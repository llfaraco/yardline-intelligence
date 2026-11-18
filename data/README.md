# Data

This directory documents the local data layout. Raw datasets are not stored in Git.

## Source families reported by the local v0.4 pipeline

- nflverse play-by-play;
- weekly snap counts;
- injury reports;
- schedules and game metadata;
- DynastyProcess player identifier mappings.

The reported canonical player join is GSIS ↔ Sleeper through the identifier mapping table, not fuzzy matching on player names.

## Suggested local layout

```text
data/
├── raw/
│   ├── nflverse/
│   └── player_ids/
├── interim/
├── processed/
│   ├── player_week/
│   └── features/
└── catalog/
    └── source_manifest.json
```

Confirm exact filenames and paths against the migrated ingestion code before running a pipeline.

## Rules

- Do not commit raw Parquet/CSV files or private league exports.
- Record provider, source URL, retrieval timestamp, season, row count and SHA-256 checksum.
- Preserve original source files and write transformations separately.
- Keep unavailable values null; do not silently turn missing into zero.
- Document filters, joins, deduplication and scoring transformations.
- Review provider terms before redistributing any data.
