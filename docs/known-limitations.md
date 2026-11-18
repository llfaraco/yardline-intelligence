# Known limitations and migration audit

This document separates implemented code from claims that still require independent verification.

## 1. Temporal split versus model fitting

The pipeline constructs a training dataframe for 2018–2023, a selection dataframe for 2024 and a test dataframe for 2025. In the current v0.4.1 ablation code, however, `ExpandedFeatureStore` is created from `df_select`, and candidate models are fitted on that store. The 2018–2023 dataframe is not passed to those model `fit` calls.

**Impact:** the archived metadata describes the intended split, but the current model-fitting path does not implement that split as stated.

**Required correction:** fit candidate models on the historical training pool, select hyperparameters using 2024, then evaluate once on 2025. Rebuild the feature-store API to keep each player's history season-aware during multi-season fitting. Archive prediction-level outputs and rerun the experiment.

## 2. Multi-season rolling features

The original vectorized feature engineering grouped by player ID only. When seasons are concatenated, rolling windows could cross season boundaries. The migration changes the grouping key to player plus season and adds `tests/test_feature_engineering.py`.

**Status:** code change and regression test added; not independently executed after migration.

## 3. Missing-value treatment

Some model helpers convert absent feature values to `0.0`. This can conflate missing information with an observed zero and conflicts with the project's intended missing-data policy.

**Required correction:** use a documented imputation strategy fitted on training data only, add missingness indicators, and test the behavior with controlled fixtures.

## 4. Probabilistic evaluation

The latest archived v0.4.1 JSON reports `probabilistic.n_residuals = 0`. The code contains residual-distribution and interval routines, but the archived run does not demonstrate that a usable residual pool was created.

The current CRPS-labelled calculations are interval-based approximations, not a validated implementation of the formal Continuous Ranked Probability Score.

**Required correction:** generate genuine out-of-sample residuals, compute proper probabilistic scores, and evaluate empirical interval coverage by season/week and projection band.

## 5. Adaptive Ridge comparison label

The archived JSON reports Adaptive Ridge MAE 2.6619 and stores a comparison baseline MAE of 2.6973 (SMA-3), yielding the reported +1.31%. SMA-5 has MAE 2.6249 and remains better. The 1.31% value must not be described as an improvement over SMA-5.

## 6. Reproduction status

The original development handoff reported 12 passing tests. The migration adds one season-boundary regression test and isolates Parquet test output under `tmp_path`. The complete suite has not yet been run from a clean clone of this repository.

## 7. Data redistribution

The original branch contained approximately 233 MB of raw caches, local DuckDB data, Parquet files and Python bytecode. These are not copied into this public repository. Raw datasets can be downloaded from the documented public providers; generated local files are ignored by Git. Raw experiment JSON and reports are retained.

## Release standard

Do not describe this project as production-ready or the v0.4.1 experiment as independently reproduced until the temporal fitting path, missing-data policy, probabilistic evaluation and clean-environment test run have been addressed.
