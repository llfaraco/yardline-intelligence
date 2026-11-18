# Data governance and reproducibility

## Source provenance

The ingestion layer records provider/source, URL, season, retrieval timestamp, row count and SHA-256 checksum in a DuckDB source catalog. Experiment records should also include the code revision, source checksums, scoring rules, temporal cutoffs, model parameters and random seeds.

## Identity resolution

Use stable player identifiers and explicit mapping tables. The research pipeline maps GSIS IDs to Sleeper IDs through the DynastyProcess player-ID table. Do not join player records by display name alone.

## Privacy and secrets

- Never commit API tokens, SSH keys, passwords, cookies or private environment files.
- Do not publish private Sleeper league identifiers, rosters or manager information without authorization.
- Redact user-level identifiers from example snapshots.
- Keep secrets in local environment variables or an approved secret manager.

## Dataset redistribution

This repository contains code, documentation and small experiment artifacts. Full raw nflverse CSV caches, local DuckDB databases and generated Parquet files are excluded. The data-preparation script downloads sources from their public providers. Review provider terms before redistributing data outside this workflow.

## Local storage

- `data/cache/`: downloaded CSV/CSV.GZ sources (ignored).
- `data/parquet/`: player-week datasets by season (ignored).
- `data/intelligence.duckdb`: local source catalog (ignored).
- `data/experiments/`: versioned experiment JSON and reports (tracked).

## Reproducibility

A reproducible experiment requires source checksums, dependency versions, code revision, scoring rules, temporal split, random seeds, model parameters and prediction-level outputs. A README table alone is not a reproducibility artifact.
