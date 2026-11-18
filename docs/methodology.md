# Modeling methodology

## Prediction target

Weekly wide-receiver (WR) fantasy production under the Yardline Half-PPR scoring preset. The scoring transformation is defined in `yardline_intelligence.config.HALF_PPR` and should be recorded with every experiment.

## Intended temporal protocol

| Role | Seasons |
|---|---|
| Training pool | 2018–2023 |
| Model selection | 2024 |
| Retrospective test | 2025 |

The protocol is chronological, not a random split. Features for a target week must use only information available before the prediction cutoff.

**Current implementation caveat:** the latest pipeline still fits candidate models using `store_select`, built from the 2024 selection dataframe. The historical `df_train` is prepared but is not passed into the candidate model `fit` calls. Therefore, the latest reported results do not yet implement the declared 2018–2023 training pool. A correct fix must make the feature store and baseline lookup season-aware and process Context Engine inputs per historical season.

## Feature families

- **Player usage and production:** snap share, targets, air yards, EPA, receptions, receiving/rushing yards, games played, trends and consistency.
- **Availability and injuries:** games missed, weeks injured, limited practice, weekly status, recurring injury signals and weeks since return.
- **Quarterback:** EPA per dropback, CPOE, passing volume/efficiency, pressure and starter changes.
- **Offensive line and defense:** pressure allowed, sack rate, passing EPA, opponent passing EPA, passing yards allowed and explosive plays.
- **Game script:** offensive pace, pass rate, home field, rest, market total/spread and implied points. Market data requires source and timestamp.

Contextual features are now included by `ExpandedFeatureStore.features_for()` when present in the target row. The owner reports that ablations C (injuries) and D (injuries + QB + OL) produce different results from B, confirming feature wiring. This does not establish that they improve accuracy: reported C MAE is 2.6967 and D MAE is 2.7167, both above SMA-5 at 2.6249.

## Missing data and athlete profiles

The model code still contains paths that coerce absent feature values to zero. The research protocol should distinguish missing information from observed zero and fit imputation rules on training data only.

The current “rookies” segment uses `weeks_active <= 2`; it is a short-history proxy rather than an official rookie-season label.

## Baselines and model selection

Simple moving averages are mandatory reference models. The latest owner-reported SMA-5 result is MAE 2.6249 and Spearman 0.5666. Adaptive Ridge is reported at MAE 2.67 / Spearman 0.562 and does not beat SMA-5. The reported injury and QB/OL ablations also do not beat SMA-5.

A complex model is not promoted merely because it is more sophisticated. Comparisons must use identical observations and a declared, correctly implemented temporal training protocol.

## Confidence intervals

The pipeline now computes cluster-bootstrap intervals for baseline and residual model outputs. The owner reports an MAE interval of [2.577, 2.774] during isolated validation. The latest run artifact must identify the model, sample, cluster unit and confidence level for this interval to be independently interpreted.

## Probabilistic outputs

The latest run reports 3,509 out-of-sample residuals and an empirical residual distribution, including when SMA-5 is the champion. This is not yet a validated calibration result:
- the pipeline does not invoke the calibration evaluator in the run path;
- the forecast helper currently maps the 5th/95th residual percentiles to fields named P10/P90;
- the CRPS-labelled metric is an interval-based approximation, not formal CRPS.

## Scientific reporting standard

Preserve negative results. Do not claim improvement unless raw out-of-sample evidence supports it. Report sample counts, prediction cutoffs, source checksums, model parameters, confidence intervals, prediction-level outputs and the exact code revision.
