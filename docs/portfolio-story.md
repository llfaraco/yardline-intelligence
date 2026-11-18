# Portfolio case study — Yardline Intelligence

## One-line summary

A temporal machine-learning research pipeline that tests whether contextual NFL data improves weekly wide-receiver fantasy-point forecasts beyond simple historical baselines.

## The problem

Fantasy football decisions require forecasts under uncertainty. Recent production matters, but it is not the whole story: opportunity, injuries, quarterback play, offensive line performance, defensive matchups and game environment can all influence a weekly outcome.

The challenge is to turn those signals into a forecast without leaking information from the future or confusing model complexity with predictive value.

## The data science work

The reported project combines multi-season NFL play-by-play and weekly usage data, player identity resolution through stable identifiers, temporal feature engineering, contextual feature families, baseline/residual/adaptive regression models, walk-forward evaluation and decision-oriented metrics.

## The scientific challenge

The most important finding so far is not that a complex model won. The reported v0.4 benchmark still places SMA-5 ahead on MAE. That is a useful research result: it prevents a more complex model from being presented as better without evidence.

The next task is to reproduce the reported experiment from a clean public repository and determine whether the adaptive model's comparison label is correct.

## Skills demonstrated

Python, data engineering, temporal data modeling, feature engineering, statistical learning, regularization, LightGBM/XGBoost experimentation, time-aware validation, leakage prevention, model evaluation, error analysis and experiment documentation.

## Evidence standard

Expand this case study with plots and claims only after source code and raw experiment artifacts have been migrated and reproduced. Distinguish reported local results from independently reproduced results.
