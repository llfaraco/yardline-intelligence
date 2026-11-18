# Tests

The migrated source test suite covers:

- Data Foundation schema/catalog behavior and Parquet round trips.
- Temporal Feature Store history windows and leakage boundaries.
- WR model fitting and non-negative predictions.
- Season-boundary isolation for vectorized temporal features.

The original development handoff reported 12 passing tests before migration. This repository adds a season-boundary regression test and isolates Parquet test output under pytest's temporary directory. The full suite must be run from a clean environment before the migrated project is considered verified.

Run:

```bash
python -m pytest tests/ -q
```

Scientific tests still required:

- model fitting uses only the declared historical training seasons;
- selection and test seasons are disjoint from model fitting;
- player-season-week keys are unique and validated;
- canonical identifier joins are used instead of player names;
- missing-value handling does not silently turn unavailable data into observed zeroes;
- scoring matches the documented Half-PPR rules;
- probabilistic metrics use valid out-of-sample residuals and proper scoring rules.
