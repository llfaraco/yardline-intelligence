# Source migration notes

## Source of truth

- Source repository: `llfaraco/yardline`
- Source branch: `claude/yardline-intelligence-v0.1`
- Source commit: `603d89b1b31051638a11bfe49531fa51a07483aa`
- Source directory: `claude/yardline-intelligence-v0.1/`
- Migrated project version: v0.4.1

## Migration changes

- Moved the project into the standalone `src/yardline_intelligence/` package.
- Rewrote package imports from `src.*` to `yardline_intelligence.*`.
- Moved evaluation utilities into `evaluation/`, feature engineering into `feature_engineering/`, and simulation into `simulation/`.
- Corrected the repository-root path after the package move.
- Added a dependency manifest and a data-preparation script.
- Made `ensure_seasons` check every requested source file instead of checking only play-by-play cache presence.
- Made Parquet read/write helpers accept an explicit directory and isolated the data-foundation test output.
- Made temporal feature engineering group by player and season when a season column is available.
- Preserved raw experiment JSON and the original v0.4 Markdown report.

## Excluded generated files

The source subtree contained about 233 MB of files, including raw CSV caches, Parquet datasets, a local DuckDB database and Python bytecode. Those files were not migrated:

- `data/cache/`
- `data/parquet/`
- `data/intelligence.duckdb`
- `__pycache__/`

This keeps the public repository source-focused and avoids redistributing large third-party datasets. The public preparation script rebuilds local caches and Parquet datasets from the documented providers.

## History preservation

The original source commit SHA and experiment artifacts are recorded here. The new repository contains a clean snapshot import; the original Yardline repository's Git commit graph is not rewritten or deleted.
