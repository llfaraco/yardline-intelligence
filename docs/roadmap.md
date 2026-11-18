# Roadmap

## Phase 0 — Public research repository

- [x] Create the public repository and standalone package.
- [x] Establish research scope and scientific reporting rules.
- [x] Migrate the v0.4.1 source modules and experiment history.
- [x] Integrate the 2026-11-18 source follow-up commit (CI, athlete profiles, residual pool and VolumeModel correction).
- [x] Document owner-reported latest results and remaining caveats.
- [x] Add dependency metadata and data preparation script.
- [x] Add migration-specific package and season-boundary tests.
- [ ] Run all 16 standalone tests in GitHub Actions.
- [ ] Commit the latest post-fix experiment JSON and prediction-level outputs.

## Phase 1 — Reproducible temporal baseline

- [ ] Fit candidate models on the declared 2018–2023 training pool.
- [ ] Use 2024 strictly for model selection.
- [ ] Keep 2025 untouched for final evaluation.
- [ ] Make ExpandedFeatureStore row/history and baseline lookups season-aware.
- [ ] Process Context Engine features independently for each historical training season.
- [ ] Validate player-season-week keys, Half-PPR scoring and sample counts.
- [ ] Publish prediction-level outputs and recompute all headline metrics.

## Phase 2 — Context feature validation

- [ ] Audit injury/availability feature availability at prediction time.
- [ ] Validate QB context and starter-change features.
- [ ] Validate offensive line, pass rush and secondary features.
- [ ] Validate game-script timestamps and source provenance.
- [ ] Run controlled ablations against the same baseline and test sample.
- [ ] Add focused tests for contextual feature inclusion and VolumeModel P99 filtering.

## Phase 3 — Adaptive and probabilistic forecasting

- [ ] Confirm CI coverage includes ensemble group F and uses the exact prediction function for adaptive group G.
- [ ] Define train-only imputation and missingness indicators.
- [ ] Rebuild the out-of-sample residual pool from the corrected temporal protocol.
- [ ] Correct the P5/P95 to P10/P90 quantile naming mismatch.
- [ ] Replace approximate CRPS calculations with a proper scoring implementation.
- [ ] Validate calibration and decision regret on held-out observations.

## Phase 4 — Portfolio publication

- [ ] Publish reproducible notebooks.
- [ ] Add model cards and experiment registry.
- [ ] Create a concise case study for recruiters.
- [ ] Add a demo only after outputs are reproducible.

WR remains the initial modeling scope until the evaluation protocol is stable.
