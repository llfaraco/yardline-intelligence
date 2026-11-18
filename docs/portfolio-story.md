# Portfolio case study — Yardline Intelligence

## One-line summary

A temporal NFL fantasy forecasting research project that tests whether player usage and game context improve weekly wide-receiver projections beyond moving-average baselines.

## The problem

Fantasy football decisions require forecasts under uncertainty. Recent production matters, but so do opportunity, injuries, quarterback play, offensive-line performance, defensive matchups and game environment.

The challenge is to convert those signals into a forecast without leaking future information or confusing model complexity with predictive value.

## Data science work

- Multi-season nflverse play-by-play, snap counts, injury reports and game metadata.
- Canonical player identity resolution through GSIS/Sleeper IDs.
- Player-season-week data modeling and Parquet storage.
- Temporal rolling features and contextual feature families.
- Baseline, Ridge residual, adaptive and gradient-boosting experiments.
- Error metrics, rank metrics, cluster-bootstrap intervals, athlete-profile diagnostics and empirical residual distributions.
- Experiment records and source-provenance documentation.

## Latest reported result

The owner-reported v0.4.1 follow-up keeps SMA-5 as champion with MAE 2.6249 and Spearman 0.5666. Injury ablation C reports MAE 2.6967, QB/OL ablation D reports 2.7167, and Adaptive Ridge reports MAE 2.67 / Spearman 0.562. None beats SMA-5.

The latest run also reports 3,509 out-of-sample residuals, a cluster-bootstrap MAE interval [2.577, 2.774], and profile metrics for low-history players and veterans. The corresponding new experiment JSON still needs to be committed.

## Research integrity

The current pipeline still fits candidate models from the 2024 selection store rather than the declared 2018–2023 training pool. The empirical residual distribution is populated, but calibrated player-level intervals are not yet established. These limitations are documented in [the migration audit](known-limitations.md) and must be resolved before presenting the benchmark as a fully reproduced temporal experiment.

## Skills demonstrated

Python, data ingestion, DuckDB, Parquet, temporal data modeling, feature engineering, statistical learning, Ridge regression, LightGBM/XGBoost experimentation, time-aware validation, leakage prevention, model evaluation, error analysis and reproducible research practices.
