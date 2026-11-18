# Modeling methodology

## Prediction target

The research target is weekly wide-receiver (WR) fantasy production under Half-PPR scoring. The scoring transformation is defined in `yardline_intelligence.config.HALF_PPR` and should be recorded with every experiment.

## Intended temporal protocol

The archived experiment metadata declares this chronological split:

| Role | Seasons |
|---|---|
| Training pool | 2018–2023 |
| Model selection | 2024 |
| Retrospective test | 2025 |

No random train/test split is used for the reported retrospective experiment. Features for a target week must be computed from information available before that week's prediction cutoff.

**Implementation caveat:** the current v0.4.1 ablation path fits candidate models from the 2024 selection store, not from the combined 2018–2023 training dataframe. The declared split and the actual fitting path are therefore not yet aligned. Do not describe the model metrics as a fully validated 2018–2023 train / 2024 select / 2025 test experiment until this is corrected and rerun.

## Feature families

- **Player usage and production:** snap share, targets, air yards, EPA, receptions, receiving/rushing yards, games played, trends and consistency.
- **Availability and injuries:** games missed, weeks injured, limited practice, weekly status, recurring injury signals and weeks since return.
- **Quarterback:** EPA per dropback, CPOE, passing volume/efficiency, pressure and starter changes.
- **Offensive line and defense:** pressure allowed, sack rate, passing EPA, opponent passing EPA, passing yards allowed and explosive plays.
- **Game script:** offensive pace, pass rate, home field, rest, market total/spread and implied points. Market data requires source and timestamp.

Temporal feature engineering was updated during migration to group rolling windows by player and season when a season column is present. A regression test was added for cross-season boundary behavior. The test has not yet been executed in a clean environment.

## Missing data

Raw source absence and model-matrix imputation are different concepts. Some model feature-vector helpers currently coerce missing values to zero. This is a known limitation, not a claim that missing values are always preserved end-to-end. The next scientific revision should fit imputation rules on training data only and add missingness indicators.

## Baselines and model selection

Simple moving averages are mandatory reference models. The archived v0.4 benchmark compares SMA-3 and SMA-5 with residual Ridge, residual LightGBM and adaptive Ridge.

A complex model is not promoted merely because it is more sophisticated. It must show reliable out-of-sample improvement over the strongest baseline on identical observations.

## Metrics

- **MAE:** mean absolute forecast error, in fantasy points.
- **RMSE:** root mean squared error.
- **Bias:** signed average error; the implementation uses prediction minus actual.
- **Spearman:** rank correlation between predicted and actual outcomes.
- **Start/Sit Win Rate:** pairwise ranking accuracy as implemented in the Accuracy Lab.
- **Decision Regret:** cost of a model-selected action relative to the evaluation comparator.
- **P10/P90 coverage:** empirical coverage of predictive intervals.
- **CRPS:** the current simulation implementation uses an approximation; it should not be presented as a formal CRPS implementation until replaced and validated.

## Probabilistic outputs

The latest archived v0.4.1 experiment records `n_residuals = 0`. The probabilistic code exists, but the archived run does not establish calibrated intervals or a validated residual distribution.

## Scientific reporting standard

Preserve negative results. Do not claim an improvement unless raw out-of-sample evidence supports it. Comparisons must use identical observations and target definitions. Report sample counts, cutoffs, source checksums, model parameters, random seeds and confidence intervals where available.
