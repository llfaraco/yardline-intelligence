# Yardline Intelligence

**NFL fantasy football forecasting research | Data Science · Data Engineering · Machine Learning**

Yardline Intelligence is an independent research project exploring weekly NFL player-performance forecasts for fantasy football. It combines public NFL datasets, temporal feature engineering, contextual signals, statistical baselines, machine-learning experiments and decision-oriented evaluation.

> **Status:** research prototype, v0.4.1 source synchronized through 2026-11-18. Initial modeling scope: wide receivers (WR). The latest source corrections are integrated, but the declared historical train/selection/test protocol and probabilistic calibration still require further work.

[Português (Brasil)](README.pt-BR.md) · [Architecture](docs/architecture.md) · [Methodology](docs/methodology.md) · [Latest results](docs/experiments/v0.4-results.md) · [Known limitations](docs/known-limitations.md)

## Research question

Can temporally valid player-level and game-context features improve weekly Half-PPR fantasy forecasts beyond a simple recent-performance baseline?

The project follows a baseline-first principle: a complex model is not promoted unless it demonstrates a reliable improvement over the strongest baseline on identical held-out observations.

## Research architecture

| Layer | Responsibility |
|---|---|
| Data Foundation | Ingest play-by-play, snap counts, injury reports, games and canonical player IDs; preserve source metadata and SHA-256. |
| Temporal Feature Store | Build player-week features, rolling windows and lagged snapshots. |
| Feature Engineering | Derive usage, volume, efficiency, consistency and trend signals. |
| Context Engine | Add injury/availability, quarterback, offensive-line/defense and game-script context. |
| Modeling & Simulation | Compare SMA baselines, Ridge residual models, adaptive shrinkage, LightGBM and XGBoost; build empirical residual distributions. |
| Accuracy Lab | Measure MAE, RMSE, bias, Spearman, Start/Sit pairwise accuracy, athlete profiles and cluster-bootstrap intervals. |

## Latest v0.4.1 validation snapshot

The source branch received follow-up commit [`24c4f315`](https://github.com/llfaraco/yardline/commit/24c4f315be9852fa1afa584173ee5ded34cbfb0d) on 2026-11-18. The results below are reported by the project owner. The corresponding new experiment JSON has not yet been committed to the source branch.

| Experiment / model | MAE | Spearman | Notes |
|---|---:|---:|---|
| **SMA-5 baseline (champion)** | **2.6249** | **0.5666** | Lowest reported MAE |
| B: volume + efficiency Ridge | 2.6995 | — | Context-free residual model |
| C: injury context | 2.6967 | — | Context changes predictions, but does not beat SMA-5 |
| D: injury + QB + OL | 2.7167 | — | Context changes predictions, but does not beat SMA-5 |
| G: adaptive Ridge | 2.67 | 0.562 | Does not beat SMA-5 |

Additional reported diagnostics:

- **3,509 out-of-sample residuals** collected for the empirical probabilistic residual distribution when SMA-5 is champion.
- A cluster-bootstrap MAE interval of **[2.577, 2.774]** was reported during isolated validation. The latest JSON is needed to independently identify the exact model/sample associated with this interval.
- Low-history group labelled “rookies”: MAE 2.1466 (n=307); veteran group: MAE 2.7791 (n=1,486). In the current implementation, “rookie” is a proxy for `weeks_active <= 2`, not an official NFL rookie designation.
- Weeks 9–11 of 2025 are present (230, 225 and 247 rows respectively). The reported 994-observation gap was attributed to players without a usable prior-game baseline, not missing weeks.
- The weighted ensemble path was executed; no ensemble metric was included in the supplied summary.

The earlier v0.3/v0.4 experiment JSONs remain preserved under `data/experiments/`. See [the latest validation record](docs/experiments/v0.4-results.md).

## Important methodology caveats

The pipeline metadata declares 2018–2023 as the training pool, 2024 as selection and 2025 as test. A code review of the latest source commit still finds candidate models being fitted with `store_select` (the 2024 selection dataframe), not the combined 2018–2023 training dataframe. The latest reported metrics therefore do **not** establish the declared 2018–2023 train / 2024 select / 2025 test protocol.

The empirical residual pool is now populated, but calibrated player-level intervals are not yet established: the pipeline records the distribution without running its calibration evaluator, the forecast helper maps residual P5/P95 to fields named P10/P90, and the CRPS-labelled calculation remains an approximation.

These limitations are documented explicitly in [the migration audit](docs/known-limitations.md).

## Data sources

- [nflverse data](https://github.com/nflverse/nflverse-data): play-by-play, snap counts and injury reports.
- [nflverse games data](https://github.com/nflverse/nfldata): schedules and game metadata.
- [DynastyProcess player IDs](https://github.com/dynastyprocess/data): canonical GSIS/Sleeper identifier mapping.

Raw CSV caches, local DuckDB files and generated Parquet datasets are excluded from Git. They can be rebuilt locally using the preparation script. Experiment JSON and Markdown reports are versioned.

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

The project owner reports that the 12 original upstream tests pass after the v0.4.1 corrections. This standalone repository now contains 16 test functions: the 12 upstream tests plus four migration/regression tests. The complete suite still needs a clean CI run.

## Repository layout

```text
.
├── data/experiments/       # Archived experiment JSON and reports
├── docs/                   # Architecture, methodology, results and audit notes
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

1. Correct the historical train/selection/test fitting path and make the feature store season-aware.
2. Process Context Engine features separately for every historical training season.
3. Run the complete standalone test suite in CI and commit the latest experiment JSON.
4. Correct probabilistic quantile labels and validate calibration with proper scoring rules.
5. Expand beyond WR only after the evaluation protocol is stable.

## Relationship to Yardline Football

This repository is the independent research and portfolio project. The production Yardline Football application is a separate codebase. No model here should be considered production-ready or connected to live user leagues without separate integration and validation.

Yardline Intelligence is not an official NFL product and is not affiliated with the NFL or nflverse.
