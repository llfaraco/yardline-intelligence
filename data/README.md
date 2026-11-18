# Data and local artifacts

Raw source files and generated datasets are intentionally excluded from Git. The repository preserves experiment JSON and Markdown reports under `data/experiments/`.

## Local directory layout

```text
data/
├── cache/          # Downloaded CSV/CSV.GZ sources (ignored)
├── parquet/        # Player-season-week datasets (ignored)
├── experiments/    # Versioned experiment JSON and Markdown reports
├── models/         # Optional local model artifacts (ignored)
└── intelligence.duckdb  # Local provenance/catalog database (ignored)
```

## Source families

- nflverse play-by-play.
- nflverse snap counts.
- nflverse injury reports.
- nflverse game metadata.
- DynastyProcess player identifier mapping.

The current ingestion implementation stores season-scoped files in `data/cache/` and builds Parquet datasets in `data/parquet/<season>/`.

## Prepare local data

```bash
python scripts/prepare_data.py --seasons 2018,2019,2020,2021,2022,2023,2024,2025
```

This downloads public source data and builds player-week Parquet files. The full set is large; run only the seasons required for your experiment when possible.

## Governance

- Record source URL, season, retrieval timestamp, row count and SHA-256.
- Keep source files immutable and write transformed data separately.
- Use GSIS/Sleeper identifier mappings rather than name-based joins.
- Do not commit private league exports, credentials or user-level information.
- Review provider terms before redistributing source datasets.
- Missing source values must remain distinguishable from observed zeroes.
