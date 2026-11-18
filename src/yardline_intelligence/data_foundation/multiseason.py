"""Camada 1 estendida — Data Foundation com DuckDB e Parquet particionado.

Fornece ingestão otimizada de múltiplas temporadas com:
- Cache incremental de features (não reprocessar temporadas completas)
- Parquet particionado por temporada e semana
- Controle de consumo de memória e tempo de execução
- Rastreabilidade completa (hash SHA-256, timestamp, número de linhas)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .ingest import (
    connect_db, init_schema, SourceCatalog, ingest_season, read_csv_cached,
    CACHE_DIR, SOURCES,
)


def ingest_multiple_seasons(seasons: list[int],
                            sources: Optional[list[str]] = None,
                            db_path: Optional[Path] = None) -> dict:
    """Baixa e registra as fontes de várias temporadas.

    Retorna dicionário temporada -> {source: resultado}.
    """
    db = connect_db(db_path)
    try:
        init_schema(db)
        catalog = SourceCatalog(db)
        results: dict[int, dict] = {}
        for season in seasons:
            results[season] = ingest_season(db, season, sources=sources, catalog=catalog)
        return results
    finally:
        db.close()


def load_multi_season_dataset(seasons: list[int],
                              parquet_dir: Optional[Path] = None) -> pd.DataFrame:
    """Carrega e concatena os datasets parquet de várias temporadas."""
    from ..config import PARQUET_DIR
    parquet_dir = parquet_dir or PARQUET_DIR
    frames = []
    for season in seasons:
        path = parquet_dir / str(season) / f"dataset_{season}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def available_seasons(cache_dir: Optional[Path] = None) -> list[int]:
    """Retorna as temporadas que já têm pbp em cache."""
    cache_dir = cache_dir or CACHE_DIR
    seasons = []
    for p in cache_dir.glob("pbp_*.csv.gz"):
        try:
            # Nome do arquivo: pbp_<temporada>.csv.gz
            seasons.append(int(p.name.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return sorted(seasons)


def ensure_seasons(seasons: list[int],
                   db_path: Optional[Path] = None,
                   sources: Optional[list[str]] = None) -> dict:
    """Ensure every requested source file exists for each season.

    The previous implementation checked only play-by-play files. A partially
    populated cache could therefore skip missing injuries, snaps, IDs or games.
    """
    from .ingest import CACHE_DIR, connect_db, init_schema, SourceCatalog

    wanted = sources or list(SOURCES.keys())
    db = connect_db(db_path)
    try:
        init_schema(db)
        catalog = SourceCatalog(db)
        results: dict[int, dict] = {}
        for season in seasons:
            missing_sources = []
            for name in wanted:
                extension = ".csv.gz" if name == "pbp" else ".csv"
                path = CACHE_DIR / f"{name}_{season}{extension}"
                if not path.exists():
                    missing_sources.append(name)
            if missing_sources:
                results[season] = ingest_season(
                    db, season, sources=missing_sources, catalog=catalog
                )
            else:
                results[season] = {
                    name: {"status": "cached"} for name in wanted
                }
        return results
    finally:
        db.close()


def summary_catalog(db_path: Optional[Path] = None) -> list[dict]:
    """Resumo de todas as fontes já ingeridas no catálogo."""
    db = connect_db(db_path)
    try:
        init_schema(db)
        rows = db.execute("""
            SELECT source, season, rows, captured_at, content_sha256
            FROM source_catalog
            ORDER BY season, source
        """).fetchall()
        return [dict(zip(["source", "season", "rows", "captured_at", "sha256"], r)) for r in rows]
    finally:
        db.close()


def build_dataset_parquet(season: int,
                          cache_dir: Optional[Path] = None,
                          parquet_dir: Optional[Path] = None,
                          force_rebuild: bool = False) -> Path:
    """Constrói e salva o dataset parquet de uma temporada se ainda não existir.

    Usa cache incremental: se o parquet já existir e não for forçado, pula a construção.
    """
    from ..data_foundation.dataset import build_player_dataset
    from ..config import PARQUET_DIR

    cache_dir = cache_dir or CACHE_DIR
    parquet_dir = parquet_dir or PARQUET_DIR

    out_path = parquet_dir / str(season) / f"dataset_{season}.parquet"

    if out_path.exists() and not force_rebuild:
        return out_path

    # Constrói o dataset base (sem features contextuais, que são pesadas)
    df = build_player_dataset(season, cache_dir=cache_dir, parquet_dir=parquet_dir)
    if df.empty:
        return out_path

    # Salva em parquet particionado
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return out_path


def build_all_seasons_parquet(seasons: list[int],
                              cache_dir: Optional[Path] = None,
                              parquet_dir: Optional[Path] = None,
                              force_rebuild: bool = False) -> dict:
    """Constrói parquets para múltiplas temporadas com cache incremental."""
    cache_dir = cache_dir or CACHE_DIR
    parquet_dir = parquet_dir or PARQUET_DIR

    results: dict[int, dict] = {}
    for season in seasons:
        started = pd.Timestamp.now().isoformat()
        try:
            out_path = build_dataset_parquet(season, cache_dir=cache_dir,
                                             parquet_dir=parquet_dir,
                                             force_rebuild=force_rebuild)
            results[season] = {
                "status": "ok",
                "path": str(out_path),
                "exists": out_path.exists(),
            }
        except Exception as exc:
            results[season] = {
                "status": "error",
                "message": str(exc),
            }

    return results
