# Source migration notes

## Original source import

- Source repository: `llfaraco/yardline`
- Source branch: `claude/yardline-intelligence-v0.1`
- Initial source commit: `603d89b1b31051638a11bfe49531fa51a07483aa`
- Source directory: `claude/yardline-intelligence-v0.1/`
- Migrated project version: v0.4.1

## Follow-up source synchronization (2026-11-18)

The standalone migration was synchronized with source commit [`24c4f315be9852fa1afa584173ee5ded34cbfb0d`](https://github.com/llfaraco/yardline/commit/24c4f315be9852fa1afa584173ee5ded34cbfb0d).

Files updated from that commit:
- `src/yardline_intelligence/pipeline.py`: cluster-bootstrap CI for baseline and model groups, athlete-profile metrics, and residual collection from SMA-5 when the champion is a pure baseline.
- `src/yardline_intelligence/models/wr_models.py`: P99 target outlier filtering and alpha 10.0 regularization in the WR volume model.

The contextual feature wiring in `expanded_store.py` and baseline Spearman calculation in `eval_vectorized.py` were already present in the initial source commit and were included in the first migration.

The source owner reports that the 12 original tests pass. The standalone repository contains 16 test functions, including four migration/regression tests; all 16 still require a clean CI run.

## Migration changes

- Reorganized source into the installable `src/yardline_intelligence/` package.
- Rewrote package imports from `src.*` to `yardline_intelligence.*`.
- Moved evaluation utilities into `evaluation/`, feature engineering into `feature_engineering/`, and simulation into `simulation/`.
- Corrected the repository-root path after the package move.
- Added dependency metadata and a data-preparation script.
- Made `ensure_seasons` check every requested source file instead of checking only play-by-play cache presence.
- Made Parquet read/write helpers accept an explicit directory and isolated the data-foundation test output.
- Made temporal feature engineering group by player and season when a season column is present.
- Preserved original v0.3/v0.4 experiment JSON and the original report.

## Excluded generated files

The source subtree contained approximately 233 MB of raw caches, local DuckDB data, Parquet files and Python bytecode. These were not migrated:
- `data/cache/`
- `data/parquet/`
- `data/intelligence.duckdb`
- `__pycache__/`

## Current research caveat

The synchronized pipeline still fits candidate models from the 2024 `store_select`, not the declared 2018–2023 `df_train`. The temporal split metadata and actual fit path are therefore not aligned. See `docs/known-limitations.md`.

The latest post-fix metrics were supplied in the source update, but the corresponding new experiment JSON is not present in the source branch tree. The old experiment artifacts remain unchanged and are not evidence for the newer numbers.

## History preservation

The source commit SHAs and original experiment artifacts are recorded here. The original Yardline repository's Git commit graph is not rewritten or deleted.
