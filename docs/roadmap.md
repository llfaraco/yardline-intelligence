# Roadmap

## Phase 0 — Public research repository

- [x] Create the public repository.
- [x] Establish research scope and scientific reporting rules.
- [x] Document the reported v0.4 benchmark and its caveats.
- [ ] Migrate source code, tests and raw experiment artifacts.
- [ ] Pin dependencies and verify clean installation.

## Phase 1 — Reproducible baseline

- [ ] Reproduce SMA-3 and SMA-5 from source datasets.
- [ ] Validate player-week keys and Half-PPR target calculation.
- [ ] Add temporal leakage tests for rolling features.
- [ ] Publish prediction-level output for the 2025 test window.
- [ ] Recompute MAE, RMSE, Bias and Start/Sit metrics.

## Phase 2 — Context feature validation

- [ ] Audit injury/availability feature availability at prediction time.
- [ ] Validate QB context and starter-change features.
- [ ] Validate offensive line, pass rush and secondary features.
- [ ] Validate game-script timestamps and source provenance.
- [ ] Run controlled ablations against the same baseline and test sample.

## Phase 3 — Adaptive and probabilistic forecasting

- [ ] Reconcile the Adaptive Ridge comparison label.
- [ ] Evaluate shrinkage and confidence adaptation.
- [ ] Validate quantile regression and interval coverage.
- [ ] Evaluate predictive distributions using CRPS.
- [ ] Assess calibration and decision regret.

## Phase 4 — Portfolio publication

- [ ] Publish reproducible notebooks.
- [ ] Add architecture and model diagrams.
- [ ] Add model cards and experiment registry.
- [ ] Create a concise case study for recruiters.
- [ ] Add a demo only after outputs are reproducible.

WR remains the initial modeling scope until the evaluation protocol is stable.
