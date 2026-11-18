# Portfolio case study — Yardline Intelligence

## One-line summary

A temporal NFL fantasy forecasting research project that tests whether player usage and game context can improve weekly wide-receiver projections beyond simple moving-average baselines.

## The problem

Fantasy football decisions require forecasts under uncertainty. Recent production matters, but it is not the whole story: opportunity, injuries, quarterback play, offensive-line performance, defensive matchups and game environment can all influence a weekly outcome.

The challenge is to turn those signals into a forecast without leaking future information or confusing model complexity with predictive value.

## Data science work

The project brings together:

- Multi-season nflverse play-by-play, snap counts, injury reports and game metadata.
- Canonical player identity resolution through GSIS/Sleeper IDs.
- Player-season-week data modeling and Parquet storage.
- Temporal rolling features and contextual feature families.
- Baseline, Ridge residual, adaptive and gradient-boosting experiments.
- Error metrics, ranking metrics, confidence intervals and experiment records.

## The scientific result

The archived v0.4.1 result keeps SMA-5 as champion with MAE 2.6249. Adaptive Ridge improves over SMA-3 but does not beat SMA-5. The project treats that as a meaningful result rather than presenting the most complex model as the winner.

The migration review also found methodological issues that must be resolved before the headline benchmark can be considered independently reproducible. Those issues are documented in [known limitations](known-limitations.md).

## Skills demonstrated

Python, data ingestion, DuckDB, Parquet, temporal data modeling, feature engineering, statistical learning, Ridge regression, LightGBM/XGBoost experimentation, time-aware validation, leakage prevention, model evaluation, error analysis and reproducible research practices.

## Evidence standard

Use archived numbers only as reported local results until the pipeline is corrected, tests are rerun from a clean environment and the benchmark is independently reproduced.
