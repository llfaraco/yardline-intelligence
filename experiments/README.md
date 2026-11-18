# Experiments

Each experiment should be immutable and traceable.

## Recommended contents

- configuration file (YAML/JSON);
- source data manifest and checksums;
- code commit SHA;
- dependency/environment lock information;
- prediction-level output for the evaluation sample;
- aggregate metrics and subgroup diagnostics;
- experiment notes, including failed hypotheses.

Use descriptive IDs, for example `2025-wr-half-ppr-v04-adaptive-ridge`. Do not overwrite prior runs. Store small, non-sensitive metadata and metrics in Git; keep large datasets and private exports outside the repository.

The original v0.4 experiment artifacts have not yet been migrated.
