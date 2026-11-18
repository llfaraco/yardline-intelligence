# Yardline Intelligence

**NFL player performance forecasting | Data Science · Statistical Modeling · Machine Learning**

Yardline Intelligence is a research project focused on forecasting weekly NFL player performance for fantasy football decision support. The project investigates whether historical production, player usage, availability, quarterback/offensive-line context, defensive matchups and game environment can improve predictions beyond simple historical baselines.

> **Project status:** research pipeline v0.4 has been developed and evaluated locally. This repository is being established as the reproducible, public portfolio home for the project. The original implementation and experiment artifacts still need to be migrated from the isolated development workspace. Until that migration and a clean reproduction are complete, the results below are reported project results, not independently reproduced by this repository.

## Research question

Can temporally valid player-level and game-context features improve weekly Half-PPR fantasy-point forecasts over a simple moving-average baseline?

The initial modeling scope is **wide receivers (WR)** and weekly player-level forecasts. The project deliberately prioritizes honest out-of-sample evaluation over model complexity.

## Reported v0.4 benchmark

Retrospective evaluation on the 2025 season, as reported by the local v0.4 experiment:

| Model | Observations | MAE | RMSE | Bias | Spearman | Start/Sit |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline SMA-5** | 3,509 | **2.6249** | 3.5757 | -0.0601 | — | 69.98% |
| Baseline SMA-3 | 3,509 | 2.6973 | 3.7075 | -0.0271 | — | 69.48% |
| Residual Ridge | 3,509 | 2.6995 | 3.6855 | 0.1553 | 0.5524 | 69.43% |
| Residual LightGBM | 3,509 | 2.7468 | 3.7422 | 0.1664 | 0.5354 | 68.75% |
| **Adaptive Ridge** | 3,509 | 2.6619 | 3.6520 | 0.0157 | **0.5598** | **69.70%** |

**Interpretation:** SMA-5 remains the lowest-MAE model in the reported benchmark. The adaptive Ridge model has a higher MAE (2.6619) than SMA-5 (2.6249), despite the original experiment summary labeling its comparison as “+1.31%”. That direction/percentage discrepancy must be reconciled against the raw experiment output before describing the adaptive model as an improvement. No model gain is claimed here.

The reported evaluation contains 3,509 player-week observations from weeks 4–10 of the 2025 season. The experiment used a strict temporal split: training on 2018–2023, hyperparameter selection on 2024, and an untouched retrospective test on 2025.

See [the v0.4 experiment report](docs/experiments/v0.4-results.md) for the detailed interpretation and caveats.

## Methodological principles

- **Temporal validation:** train, selection and test periods are separated chronologically.
- **No data leakage:** features for a target week use information available before that week.
- **Baseline first:** complex models must demonstrate out-of-sample value over SMA baselines.
- **Missing is not zero:** unavailable values remain null/ missing and are not silently converted to zero.
- **Traceable data:** preserve source, capture time, row counts and SHA-256 metadata where available.
- **Canonical player identity:** use stable identifiers (GSIS/Sleeper mapping through DynastyProcess data), not name matching.
- **Reproducibility:** retain experiment configurations, metrics, data provenance and environment details.

## Research architecture

The intended pipeline is organized into six layers:

1. **Data Foundation** — ingestion, validation, source catalog and Parquet storage.
2. **Temporal Feature Store** — player-season-week features, rolling windows and lagged trends.
3. **Feature Engineering** — usage, volume, efficiency, consistency and matchup signals.
4. **Context Engine** — injuries/availability, quarterback, offensive line/pass rush/secondary and game script.
5. **Modeling & Simulation** — baselines, residual/adaptive models, quantile estimates and predictive distributions.
6. **Accuracy Lab** — temporal backtests, MAE/RMSE/Bias, rank correlation, Start/Sit metrics, decision regret, interval coverage and CRPS.

The repository structure and documentation are being established first. Source modules, tests and raw experiment artifacts will be migrated in a separate, auditable step.

## Data sources

The local research pipeline has used authorized public datasets from the nflverse ecosystem, including play-by-play, snap counts, injury reports, schedules/games and player identifier mappings. Raw datasets are intentionally **not committed** to this repository. See [data documentation](data/README.md) for the expected local layout and provenance requirements.

## Repository structure

```text
.
├── data/                 # Local datasets; raw files are not versioned
├── docs/                 # Architecture, methodology and experiment reports
├── experiments/          # Reproducible experiment configurations and outputs
├── notebooks/            # Exploratory analysis and research notebooks
├── reports/              # Generated evaluation reports and figures
├── src/yardline_intelligence/
│   ├── data_foundation/
│   ├── temporal_feature_store/
│   ├── feature_engineering/
│   ├── context_engine/
│   ├── models/
│   ├── simulation/
│   └── evaluation/
└── tests/
```

## Reproducibility status

The original local environment was reported as Python 3.15 with DuckDB, pandas, PyArrow, scikit-learn, LightGBM, XGBoost and pytest. Exact dependency versions, source code and experiment files must be migrated before a clean environment can reproduce the benchmark.

No claim of a fully reproducible public release is made until:
1. source code and tests are present in this repository;
2. dependency versions and data acquisition steps are documented;
3. the experiment can be rerun from a clean environment;
4. generated metrics match the committed experiment record.

## Roadmap

- [ ] Migrate the v0.4 source code and test suite from the isolated development workspace.
- [ ] Pin and document the Python environment and dependencies.
- [ ] Add data acquisition and integrity checks.
- [ ] Reproduce the 2018–2023 / 2024 / 2025 walk-forward experiment.
- [ ] Reconcile the adaptive Ridge comparison percentage against raw outputs.
- [ ] Add model diagnostics, calibration and probabilistic evaluation.
- [ ] Publish notebooks and visual reports with clear provenance.
- [ ] Extend the research beyond WR only after the WR evaluation protocol is stable.

## Language

The primary technical documentation is written in English for portfolio visibility. Portuguese documentation may be added where it improves explanation and accessibility.

---

**Yardline Intelligence** is an independent research project. It is not an official NFL product and is not affiliated with the NFL or nflverse.
