"""Testes da camada 1 — Data Foundation (ingestão, catálogo de fontes, rastreabilidade)."""

from __future__ import annotations

import duckdb
import pandas as pd

from yardline_intelligence.data_foundation.ingest import (
    connect_db, init_schema, SourceCatalog, ingest_season, to_parquet,
    load_parquet,
)
from yardline_intelligence.config import DATA_DIR, CACHE_DIR


def test_init_schema_creates_tables(tmp_path):
    db = connect_db(tmp_path / "t.duckdb")
    try:
        init_schema(db)
        tables = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
        for t in ("source_catalog", "ingestion_log", "player_canonical"):
            assert t in tables, f"tabela {t} ausente"
    finally:
        db.close()


def test_source_catalog_register_and_last_capture(tmp_path):
    db = connect_db(tmp_path / "t.duckdb")
    try:
        init_schema(db)
        cat = SourceCatalog(db)
        (tmp_path / "pbp_2026.csv.gz").write_bytes(b"")
        cat.register("pbp", 2026, tmp_path / "pbp_2026.csv.gz",
                     "https://example.com/pbp_2026.csv.gz", 100)
        hit = cat.last_capture("pbp", 2026)
        assert hit is not None
        assert hit["rows"] == 100
        assert hit["content_sha256"]
        assert hit["captured_at"]
    finally:
        db.close()


def test_to_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = to_parquet(df, "teste", 2026, parquet_dir=tmp_path / "parquet")
    assert path.exists()
    back = load_parquet("teste", 2026, parquet_dir=tmp_path / "parquet")
    assert len(back) == 2
    assert list(back.columns) == ["a", "b"]
