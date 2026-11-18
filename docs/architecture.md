# Architecture

## Overview

Yardline Intelligence is a temporal forecasting research pipeline. Its primary observation is a player-season-week record, initially focused on wide receivers (WR) and Half-PPR fantasy points.

```text
Public NFL data sources
        |
        v
1. Data Foundation
   ingestion -> source catalog -> validation -> Parquet
        |
        v
2. Temporal Feature Store
   player-season-week -> lagged windows -> historical snapshots
        |
        v
3. Feature Engineering
   usage, volume, efficiency, trends, consistency
        |
        v
4. Context Engine
   injuries | QB | OL/pass rush/secondary | game script
        |
        v
5. Modeling & Simulation
   SMA baselines | residual/adaptive models | predictive distributions
        |
        v
6. Accuracy Lab
   temporal evaluation -> error diagnostics -> decision metrics
        |
        v
Versioned experiment JSON + report
```

## Implemented package map

- `src/yardline_intelligence/data_foundation/`: ingestion, source catalog, player-week dataset construction and multi-season Parquet preparation.
- `src/yardline_intelligence/temporal_feature_store/`: per-player historical snapshots, rolling features and moving-average baselines.
- `src/yardline_intelligence/feature_engineering/`: vectorized temporal features.
- `src/yardline_intelligence/context_engine/`: injury, QB, offensive-line/defense and game-script context.
- `src/yardline_intelligence/models/`: WR volume/efficiency, direct points, residual and adaptive models.
- `src/yardline_intelligence/simulation/`: residual distributions and interval diagnostics.
- `src/yardline_intelligence/evaluation/`: core metrics, bootstrap confidence intervals and vectorized backtesting.
- `scripts/`: source-data preparation and historical diagnostic scripts.
- `tests/`: data foundation, temporal feature store, model and season-boundary tests.

## Data flow and provenance

1. Download public nflverse and DynastyProcess datasets.
2. Record source URL, capture timestamp, row count and SHA-256 in DuckDB's source catalog.
3. Build player-week datasets and store them as Parquet by season.
4. Generate lagged player features and contextual features.
5. Fit and evaluate candidate models against moving-average baselines.
6. Save experiment configuration, metrics and metadata as JSON under `data/experiments/`.

## Important implementation caveat

The v0.4.1 ablation code does not yet fit from the declared 2018–2023 training pool. It constructs that dataframe but the current candidate fitting path uses the 2024 selection store. See [known limitations](known-limitations.md). The architecture describes the intended research flow; it should not be read as proof that every step is already implemented exactly as intended.
