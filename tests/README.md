# Tests

The standalone suite contains 16 test functions:

- 12 original upstream tests, reported by the project owner as passing after the v0.4.1 source corrections.
- 4 migration/regression tests:
  - package version exposure;
  - season-boundary isolation for rolling features;
  - contextual feature propagation into `ExpandedFeatureStore.features_for()`;
  - Spearman output from the vectorized baseline evaluator.

Run:

```bash
python -m pytest tests/ -q
```

The repository now includes a GitHub Actions workflow that installs the project on Python 3.12, compiles source/tests and runs the full suite on pushes and pull requests.

The full 16-test suite has not yet been independently executed against this standalone branch. Scientific tests still required:

- model fitting uses only the declared historical training seasons;
- selection and test seasons are disjoint from model fitting;
- player-season-week keys are unique and validated;
- canonical identifier joins are used instead of player names;
- missing-value handling does not silently turn unavailable data into observed zeroes;
- scoring matches the documented Half-PPR rules;
- probabilistic metrics use valid out-of-sample residuals and proper scoring rules.
