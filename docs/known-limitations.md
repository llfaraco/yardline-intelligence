# Known limitations and migration audit

This document distinguishes implemented changes, owner-reported validation and issues still visible in the current code.

## 1. Declared temporal split versus actual model fitting

The pipeline metadata declares 2018–2023 as the training pool, 2024 as model selection and 2025 as retrospective test. The current ablation code still constructs:

- `store_select` from `df_select` (2024);
- `store_test` from `df_test` (2025);

and calls `m.fit(store_select, ...)` for the B/C/D/E/G candidate models. The combined historical `df_train` is built and passed through feature/context preparation, but it is not the dataframe used by those candidate model `fit` calls.

**Impact:** the latest reported results are not evidence of the declared 2018–2023 train / 2024 select / 2025 test protocol. They reflect candidate fitting on the 2024 selection store followed by evaluation on 2025.

**Required correction:** fit on the historical training pool, select only on 2024 and evaluate once on 2025. This requires season-aware row/history lookup in `ExpandedFeatureStore`, season-aware baseline construction and Context Engine processing separately for every training season. Do not simply concatenate seasons and reuse the current player/week-only history lookup.

## 2. Contextual features are now wired into feature snapshots

The latest migrated `ExpandedFeatureStore.features_for()` includes injury, QB, offensive-line/defense and game-script columns when present in the target row. The owner reports that ablations C and D now differ from B, confirming the columns are reaching the feature vector in the reported run.

Reported results:
- B: MAE 2.6995.
- C (injury context): MAE 2.6967.
- D (injury + QB + OL): MAE 2.7167.

These results show that the contextual ablations changed predictions, but neither C nor D beat the SMA-5 champion (MAE 2.6249).

## 3. Baseline rank metric

The vectorized baseline evaluator now computes Spearman. The latest owner-reported SMA-5 value is 0.5666. This metric is included in the latest pipeline output.

## 4. Confidence intervals

The pipeline now calculates cluster-bootstrap MAE/RMSE intervals for baseline and several residual model groups, and the owner reports a validation interval of MAE [2.577, 2.774].

Code-review items still requiring attention:
- The CI dispatch loop does not currently map ensemble group F to its prediction path, so “all models” does not include the ensemble.
- For adaptive group G, the CI/profile/probability paths can reconstruct predictions directly from the underlying estimator and may bypass the adaptive shrinkage logic used by `evaluate_adaptive`. Verify that reported uncertainty and profile metrics use the same prediction function as the headline model metric.
- The latest run JSON is not yet committed, so the interval cannot be independently tied to a named model and exact sample from the repository alone.

## 5. Probabilistic residual distribution is populated, calibration is not yet established

The latest run reports 3,509 out-of-sample residuals, replacing the previous zero-residual output. When SMA-5 is champion, the pipeline now collects residuals from that baseline and stores an empirical distribution.

This is a meaningful fix, but the current pipeline does not call `evaluate_calibration()` or generate a persisted set of per-player probabilistic forecasts as part of the run. In addition:
- `forecast_from_residuals()` maps the residual 5th and 95th percentiles to fields named P10 and P90.
- `evaluate_calibration()` labels an interval-based approximation as CRPS; it is not the formal Continuous Ranked Probability Score.

**Required correction:** define the intended quantile convention, generate player-level forecasts, evaluate coverage and proper scoring rules on held-out observations, and archive the output. Until then, describe the result as an empirical OOS residual distribution, not a calibrated probability engine.

## 6. Athlete profile labels

The pipeline defines “rookies” as players with `weeks_active <= 2` in the current feature store. This is a short-history proxy, not an official NFL rookie indicator. Use labels such as “low-history players” in external reporting unless an actual rookie-season identifier is added.

Owner-reported profile metrics:
- Low-history / “rookie” proxy: MAE 2.1466, n=307.
- Veteran group: MAE 2.7791, n=1,486.

## 7. Missing baseline observations

The owner reports that 2025 weeks 9–11 are present with 230, 225 and 247 rows. The reported 994-observation gap is attributed to players without sufficient prior-game history to create a valid moving-average baseline, rather than missing weeks. Preserve this distinction in future diagnostics and expose the number of rows excluded by each filter.

## 8. VolumeModel outlier handling

The latest source commit adds a P99 target outlier filter and stronger Ridge regularization (alpha 10.0) in the WR volume-model fit path. This change should be covered by a focused regression test and compared against the previous version on the same training/validation sample. Do not infer that it changes the headline residual-model result unless that model path is actually used in the experiment.

## 9. Test and artifact status

The project owner reports all 12 original upstream tests passing after the v0.4.1 corrections. This standalone migration currently contains 16 test functions: the 12 upstream tests plus four migration/regression tests (package version, season-boundary isolation, contextual feature propagation and baseline Spearman). The full suite has not yet been independently run in this repository.

The latest post-fix run JSON is also absent from the source branch tree. The checked-in v0.3/v0.4 JSONs are earlier artifacts and must not be presented as containing the new Spearman, CI, profile and residual-pool results.

## Release standard

Keep the PR in draft until:
1. the declared temporal training split is implemented correctly;
2. the full standalone test suite passes in CI;
3. the latest experiment JSON and prediction-level outputs are committed;
4. the probabilistic quantiles and calibration metrics are corrected and validated.
