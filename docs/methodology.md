# Modeling methodology

## Prediction target

The reported research target is weekly WR fantasy production under Half-PPR scoring. The exact scoring transformation must remain aligned with the Yardline Half-PPR preset and be included in each experiment configuration.

## Temporal evaluation protocol

The v0.3/v0.4 protocol reported by the project separates seasons chronologically:

| Role | Seasons |
|---|---|
| Training | 2018–2023 |
| Hyperparameter selection | 2024 |
| Retrospective test | 2025 |

No random train/test split is used for the headline retrospective result. A feature for a target week may only use information available before that week's prediction cutoff.

## Baselines and model selection

Simple moving averages are mandatory reference models. The reported v0.4 benchmark compares SMA-3 and SMA-5 with residual Ridge, residual LightGBM and an adaptive Ridge model.

A complex model is not promoted merely because it is more sophisticated. It must show reliable out-of-sample improvement over the baseline on the same observations and target definition.

## Feature families

- **Player usage/production:** snap share, targets, air yards, EPA, receptions, receiving/rushing yards, games played, trends and consistency.
- **Availability/injuries:** games missed, weeks injured, limited practice, weekly status, recurring injury signals and weeks since return.
- **Quarterback:** EPA per dropback, CPOE, passing volume/efficiency, pressure and starter changes.
- **OL and defense:** pressure allowed, sack rate, passing EPA, opponent passing EPA, passing yards allowed and explosive plays.
- **Game script:** offensive pace, pass rate, home field, rest, market total/spread and implied points. Market data needs source and timestamp.

## Missing data

Missing information is not equivalent to zero. Missing values remain null or are handled by an explicitly documented model-specific imputation strategy. Imputation must be fit on training data only.

## Metrics

- **MAE:** mean absolute forecast error, in fantasy points.
- **RMSE:** root mean squared error.
- **Bias:** signed average error; the sign convention must be documented in code.
- **Spearman:** rank correlation between predicted and actual outcomes.
- **Start/Sit Win Rate:** decision metric; pair construction and tie handling must be documented.
- **Decision Regret:** cost of a selected action relative to the best observed action under the evaluation setup.
- **P10/P90 coverage:** empirical coverage of predictive intervals.
- **CRPS:** quality of the full predictive distribution.

Comparisons must use identical observations. Report sample counts and confidence intervals where available.

## Scientific reporting standard

Preserve negative results. Do not claim an improvement unless raw out-of-sample evidence supports it. Link raw experiment outputs, metric definitions and code revisions by experiment ID.
