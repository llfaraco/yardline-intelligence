# Architecture

## Overview

Yardline Intelligence is a temporal forecasting pipeline. Its main unit is a player-season-week record, initially focused on wide receivers (WR) and Half-PPR fantasy points.

```text
Authorized source data
        |
        v
1. Data Foundation
   ingestion -> validation -> source manifest -> Parquet
        |
        v
2. Temporal Feature Store
   player/week keys -> lagged windows -> temporal joins
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
   SMA baselines | residual/adaptive models | quantiles
        |
        v
6. Accuracy Lab
   walk-forward backtest -> diagnostics -> decision metrics
        |
        v
Versioned experiment record
```

## Layer responsibilities

1. **Data Foundation:** acquire authorized data, validate schemas and keys, preserve provenance/checksums and store columnar datasets. DuckDB is used for analytical queries over Parquet.
2. **Temporal Feature Store:** build player-season-week rows. Rolling windows and trends must be shifted so a prediction for week t only uses information available before week t.
3. **Feature Engineering:** represent opportunity and production through snap share, targets, air yards, EPA, receptions, receiving/rushing yards, games and consistency.
4. **Context Engine:** provide pregame context from injuries/availability, QB, offensive line/pass rush/secondary and game script.
5. **Modeling & Simulation:** maintain simple moving-average baselines alongside residual/adaptive models. Derive probabilistic outputs from out-of-sample residuals.
6. **Accuracy Lab:** evaluate forecast error, calibration and decision usefulness with explicit sample counts and temporal cutoffs.

## Repository status

The original local v0.4 implementation has not yet been migrated into this public repository. These documents define the target architecture; they do not imply that every module is executable here.
