# Roadmap

## Phase 0 — Public research repository

- [x] Create the public repository.
- [x] Establish research scope and scientific reporting rules.
- [x] Document the reported v0.4 benchmark and caveats.
- [x] Migrate the v0.4.1 source modules into a standalone package.
- [x] Migrate raw experiment JSON and the original report.
- [x] Add dependency metadata and data preparation script.
- [x] Add a season-boundary feature regression test.
- [ ] Run the complete suite from a clean environment.
- [ ] Verify a clean install and end-to-end data preparation.

## Phase 1 — Reproducible baseline

- [ ] Correct the model-fitting path to use the declared 2018–2023 training pool.
- [ ] Use 2024 for model selection and keep 2025 untouched for final evaluation.
- [ ] Validate player-week keys, Half-PPR scoring and sample counts.
- [ ] Audit all rolling features for player-season isolation.
- [ ] Publish prediction-level outputs for the 2025 test window.
- [ ] Recompute MAE, RMSE, Bias and Start/Sit metrics.

## Phase 2 — Context feature validation

- [ ] Audit injury/availability feature availability at prediction time.
- [ ] Validate QB context and starter-change features.
- [ ] Validate offensive line, pass rush and secondary features.
- [ ] Validate game-script timestamps and source provenance.
- [ ] Run controlled ablations against the same baseline and test sample.

## Phase 3 — Adaptive and probabilistic forecasting

- [ ] Define train-only imputation and missingness indicators.
- [ ] Reconcile Adaptive Ridge comparison metrics.
- [ ] Rebuild the out-of-sample residual pool.
- [ ] Validate quantile regression and interval coverage.
- [ ] Replace approximate CRPS calculations with a proper scoring implementation.
- [ ] Assess calibration and decision regret.

## Phase 4 — Portfolio publication

- [ ] Publish reproducible notebooks.
- [ ] Add model cards and experiment registry.
- [ ] Create a concise case study for recruiters.
- [ ] Add a demo only after outputs are reproducible.

WR remains the initial modeling scope until the evaluation protocol is stable.
