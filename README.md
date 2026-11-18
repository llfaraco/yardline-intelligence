# Yardline Intelligence

**NFL fantasy football forecasting research | Data Science · Data Engineering · Machine Learning**

Yardline Intelligence is an independent research project exploring weekly NFL player-performance forecasts for fantasy football. It combines public NFL datasets, temporal feature engineering, contextual signals, statistical baselines, machine-learning experiments, and decision-oriented evaluation.

> **Current status:** research prototype, codebase migrated from the original Yardline development branch (v0.4.1). Initial modeling scope: wide receivers (WR). The reported benchmark is archived, but a clean end-to-end reproduction and several methodological corrections are still required before treating it as a validated model.

[Português (Brasil)](README.pt-BR.md) · [Architecture](docs/architecture.md) · [Methodology](docs/methodology.md) · [Results](docs/experiments/v0.4-results.md) · [Known limitations](docs/known-limitations.md)

## Research question

Can temporally valid player-level and game-context features improve weekly Half-PPR fantasy forecasts beyond simple historical baselines?

The project follows a baseline-first principle: a complex model is not promoted unless it demonstrates a reliable improvement on the same held-out observations.

## Research architecture

| Layer | Responsibility |
|---|---|
| Data Foundation | Ingest play-by-play, snap counts, injury reports, games and canonical player IDs; preserve source metadata and SHA-256 checksums. |
| Temporal Feature Store | Build player-season-week records, rolling windows and lagged features. |
| Feature Engineering | Derive usage, volume, efficiency, consistency and trend signals. |
| Context Engine | Add injury/availability, quarterback, offensive-line/defense and game-script context. |
| Modeling & Simulation | Compare SMA baselines, Ridge residual models, adaptive shrinkage, LightGBM and XGBoost; derive residual-based predictive distributions. |
| Accuracy Lab | Evaluate MAE, RMSE, bias, Spearman, Start/Sit pairwise accuracy, decision regret, interval coverage and bootstrap confidence intervals. |

## Reported v0.4.1 benchmark

The archived experiment contains 3,509 player-week observations from the 2025 season.

| Model | MAE | RMSE | Bias | Start/Sit win rate |
|---|---:|---:|---:|---:|
| **SMA-5 baseline (reported champion)** | **2.6249** | **3.5757** | **-0.0601** | **69.98%** |
| SMA-3 baseline | 2.6973 | 3.7075 | -0.0271 | 69.48% |
| Adaptive Ridge residual | 2.6619 | 3.6520 | 0.0157 | 69.70% |
| Residual Ridge | 2.6995 | 3.6855 | 0.1553 | 69.43% |
| Residual LightGBM | 2.7468 | 3.7422 | 0.1664 | 68.75% |

Adaptive Ridge is 1.31% better than SMA-3 according to the archived comparison field, but its MAE remains higher than SMA-5. SMA-5 therefore remains the lowest-MAE model in the archived experiment.

**Research integrity note:** these are reported local experiment results, not a benchmark independently reproduced from this repository. The raw JSON artifacts are preserved in `data/experiments/`.

## Important methodology caveats

A migration review found discrepancies that must be resolved before the benchmark is described as a clean 2018–2023 train / 2024 selection / 2025 test experiment:

1. The current ablation code creates the historical training dataframe but fits the models using the 2024 selection store.
2. The multi-season feature engineering path previously grouped rolling features by player only. The migration makes this grouping season-aware and adds a regression test, but the full suite has not yet been rerun.
3. The latest archived v0.4.1 JSON records `n_residuals = 0` for the probabilistic output. Calibrated prediction intervals are therefore not claimed.
4. Model feature-vector helpers currently coerce missing feature values to zero in some model paths. This needs an explicit, train-only imputation policy and missingness indicators.
5. The recorded Adaptive Ridge comparison percentage is relative to SMA-3, not evidence that it beats SMA-5.

See [known limitations](docs/known-limitations.md) for the full audit. These issues are documented rather than hidden.

## Data sources

- [nflverse data](https://github.com/nflverse/nflverse-data): play-by-play, snap counts and injury reports.
- [nflverse games data](https://github.com/nflverse/nfldata): schedules and game metadata.
- [DynastyProcess player IDs](https://github.com/dynastyprocess/data): canonical GSIS/Sleeper identifier mapping.

Raw CSV caches, local DuckDB files and generated Parquet datasets are excluded from Git. Use the preparation script to download public sources and rebuild local datasets. Experiment JSON and Markdown reports are versioned.

## Quick start

Python 3.12 or newer is specified in the project configuration.

### 1. Create an environment

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download data and build local Parquet datasets

The full 2018–2025 source set is substantial and may take time to download.

```bash
python scripts/prepare_data.py --seasons 2018,2019,2020,2021,2022,2023,2024,2025
```

### 3. Run the retrospective pipeline

```bash
python -m yardline_intelligence.pipeline --train-seasons 2018,2019,2020,2021,2022,2023 --select-season 2024 --test-season 2025
```

### 4. Run tests

```bash
python -m pytest tests/ -q
```

The source handoff reported 12 passing tests before migration. This repository adds a season-boundary regression test and isolates Parquet test output, but the complete suite has **not yet been independently executed after migration**.

## Repository layout

```text
.
├── data/experiments/       # Archived raw experiment JSON and report
├── docs/                   # Architecture, methodology, results and audit notes
├── experiments/            # Experiment registry documentation
├── reports/archive/        # Historical report snapshots
├── scripts/                # Data preparation and diagnostic backtests
├── src/yardline_intelligence/
│   ├── context_engine/
│   ├── data_foundation/
│   ├── evaluation/
│   ├── feature_engineering/
│   ├── models/
│   ├── simulation/
│   └── temporal_feature_store/
└── tests/
```

## Next research steps

1. Correct the train/selection/test fitting path and rerun the full temporal experiment.
2. Audit all rolling features for player-season isolation and prediction-time availability.
3. Implement explicit missing-value handling fitted on training data only.
4. Rebuild the out-of-sample residual pool and validate probabilistic calibration with proper scoring rules.
5. Recompute all metrics from prediction-level outputs and archive the complete run metadata.
6. Expand beyond WR only after the WR protocol is stable.

## Relationship to Yardline Football

This repository is the independent research and portfolio project. The production Yardline Football application is a separate codebase. No model in this repository should be considered production-ready or connected to live user leagues without separate integration and validation.

Yardline Intelligence is not an official NFL product and is not affiliated with the NFL or nflverse.
