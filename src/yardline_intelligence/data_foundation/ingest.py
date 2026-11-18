"""Camada 1 — Data Foundation.

Ingestão histórica das fontes autorizadas (nflverse play-by-play, snap counts,
injury reports, tabela de ids da DynastyProcess e jogos), conversão para Parquet
e leitura através do DuckDB. Cada arquivo baixado carrega timestamp de captura,
fonte e hash de conteúdo para rastreabilidade completa.

Regras do projeto:
- Nada é inventado. O que não existe na fonte fica ausente (null), nunca zero.
- Data leakage: uma semana só é consumível após o seu próprio cutoff; nunca
  usamos informação de semanas futuras para treinar uma semana passada.
- Identificação canônica: GSIS (play-by-play) <-> Sleeper (fantasy), nunca por nome.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from ..config import (
    CACHE_DIR, PARQUET_DIR, DATA_DIR, SOURCES, CURRENT_SEASON,
    TRAIN_SEASONS, VALIDATE_SEASON, FANTASY_POSITIONS,
)

UTC = timezone.utc


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SourceCatalog:
    """Catálogo de fontes baixadas, com metadados de rastreabilidade."""

    def __init__(self, db: duckdb.DuckDBPyConnection):
        self.db = db
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS source_catalog (
                source TEXT NOT NULL,
                season INTEGER,
                file_path TEXT NOT NULL,
                url TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                rows INTEGER NOT NULL,
                PRIMARY KEY (source, season)
            )
        """)

    def register(self, source: str, season: Optional[int], file_path: Path,
                 url: str, rows: int) -> None:
        self.db.execute("""
            INSERT OR REPLACE INTO source_catalog
            (source, season, file_path, url, captured_at, content_sha256, rows)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            source, season, str(file_path), url, now_iso(),
            sha256_of_file(file_path), int(rows),
        ])

    def last_capture(self, source: str, season: Optional[int]) -> Optional[dict]:
        row = self.db.execute("""
            SELECT source, season, file_path, url, captured_at, content_sha256, rows
            FROM source_catalog WHERE source=? AND (season IS NULL OR season=?)
            ORDER BY captured_at DESC LIMIT 1
        """, [source, season]).fetchone()
        if not row:
            return None
        return dict(zip(
            ["source", "season", "file_path", "url", "captured_at",
             "content_sha256", "rows"], row))


def connect_db(path: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    path = path or (DATA_DIR / "intelligence.duckdb")
    path.parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(path))
    db.execute("INSTALL httpfs; LOAD httpfs;") if False else None
    return db


def init_schema(db: duckdb.DuckDBPyConnection) -> None:
    """Cria as tabelas canônicas do motor (apenas metadados + dados derivados)."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS source_catalog (
            source TEXT NOT NULL,
            season INTEGER,
            file_path TEXT NOT NULL,
            url TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            rows INTEGER NOT NULL,
            PRIMARY KEY (source, season)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_log (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            season INTEGER,
            status TEXT NOT NULL,
            message TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            rows INTEGER
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS player_canonical (
            sleeper_id TEXT PRIMARY KEY,
            gsis_id TEXT,
            name TEXT NOT NULL,
            position TEXT,
            team TEXT,
            age INTEGER,
            source_url TEXT NOT NULL,
            captured_at TEXT NOT NULL
        )
    """)


def download_source(url: str, dest: Path, timeout: int = 180) -> int:
    """Baixa uma fonte para `dest` e retorna o número de linhas (aprox. para CSV)."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "yardline-intelligence/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} para {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with dest.open("wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
    # Conta linhas do CSV (subtrai o cabeçalho).
    if dest.suffix == ".gz":
        import gzip
        with gzip.open(dest, "rt", encoding="utf-8", errors="replace") as f:
            lines = sum(1 for _ in f) - 1
    else:
        with dest.open("r", encoding="utf-8", errors="replace") as f:
            lines = sum(1 for _ in f) - 1
    return max(lines, 0)


def ingest_season(db: duckdb.DuckDBPyConnection, season: int,
                  sources: Optional[list[str]] = None,
                  catalog: Optional[SourceCatalog] = None) -> dict:
    """Baixa e registra as fontes de uma temporada no cache local.

    Retorna um dicionário source -> {file, rows, captured_at, sha256}.
    """
    catalog = catalog or SourceCatalog(db)
    wanted = sources or list(SOURCES.keys())
    results: dict[str, dict] = {}
    for name in wanted:
        spec = SOURCES[name]
        url = spec.format(season=season) if "{season}" in spec else spec
        file_ext = ".csv.gz" if name == "pbp" else ".csv"
        dest = CACHE_DIR / f"{name}_{season}{file_ext}" if season \
            else CACHE_DIR / f"{name}{file_ext}"
        started = now_iso()
        try:
            rows = download_source(url, dest)
            finished = now_iso()
            catalog.register(name, season, dest, url, rows)
            db.execute("""
                INSERT OR REPLACE INTO ingestion_log
                (id, source, season, status, message, started_at, finished_at, rows)
                VALUES ((SELECT COALESCE(MAX(id),0)+1 FROM ingestion_log), ?, ?, 'ok', ?, ?, ?, ?)
            """, [name, season, f"{rows} linhas baixadas", started, finished, rows])
            results[name] = {
                "file": str(dest), "rows": rows,
                "captured_at": finished, "sha256": sha256_of_file(dest),
            }
        except Exception as exc:  # não falha a ingestão inteira por uma fonte
            finished = now_iso()
            db.execute("""
                INSERT OR REPLACE INTO ingestion_log
                (id, source, season, status, message, started_at, finished_at, rows)
                VALUES ((SELECT COALESCE(MAX(id),0)+1 FROM ingestion_log), ?, ?, 'error', ?, ?, ?, 0)
            """, [name, season, str(exc), started, finished])
            results[name] = {"error": str(exc)}
    return results


def read_csv_cached(db: duckdb.DuckDBPyConnection, path: Path,
                    columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Lê um CSV (ou .gz) do cache via DuckDB, retornando DataFrame."""
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            text = f.read()
        df = pd.read_csv(__import__("io").StringIO(text), dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if columns:
        cols = [c for c in columns if c in df.columns]
        df = df[cols]
    return df
def to_parquet(df: pd.DataFrame, name: str, season: int,
               parquet_dir: Optional[Path] = None) -> Path:
    """Persiste um DataFrame em Parquet com partição por temporada."""
    root = parquet_dir or PARQUET_DIR
    out_dir = root / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def load_parquet(name: str, season: int,
                 parquet_dir: Optional[Path] = None) -> pd.DataFrame:
    root = parquet_dir or PARQUET_DIR
    path = root / str(season) / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
