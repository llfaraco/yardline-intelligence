# Data governance and reproducibility

## Source provenance

Each ingested source should have a manifest entry containing provider/dataset name, source URL, season, retrieval timestamp, source version, file format, schema version, row count, primary-key checks and SHA-256 checksum.

The local v0.4 summary reports a source catalog with SHA-256 and capture timestamps. The public implementation should preserve that behavior during migration.

## Identity resolution

Use stable player identifiers and explicit mapping tables. The reported workflow maps GSIS to Sleeper through DynastyProcess identifiers. Do not join player records by display name alone.

## Privacy and secrets

- Never commit API tokens, SSH keys, passwords, cookies or private environment files.
- Do not publish private Sleeper league identifiers, rosters or manager information without authorization.
- Redact user-level identifiers from example snapshots.
- Keep secrets in local environment variables or an approved secret manager.

## Dataset redistribution

This repository should contain code, schemas, manifests and small non-sensitive fixtures only. Do not upload full raw nflverse datasets or third-party data unless redistribution rights are clear. Link to the provider and document acquisition steps.

## Reproducibility

A reproducible experiment requires code revision, dependency versions, source checksums, scoring rules, temporal split, random seeds, model parameters and output metrics. A README table alone is not a reproducibility artifact.
