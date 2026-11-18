# Package

The `yardline_intelligence` package contains the research pipeline migrated from the original Yardline development branch.

- `config.py`: project paths, source URLs and Half-PPR scoring.
- `pipeline.py`: historical experiment orchestration.
- `data_foundation/`: source ingestion and player-week dataset construction.
- `temporal_feature_store/`: player history and lagged feature snapshots.
- `feature_engineering/`: vectorized temporal feature generation.
- `context_engine/`: injury, QB, offensive line/defense and game-script features.
- `models/`: baseline, residual, adaptive and direct WR models.
- `simulation/`: residual distributions and interval diagnostics.
- `evaluation/`: metrics, bootstrap confidence intervals and vectorized backtests.

Run the pipeline from the repository root with:

```bash
python -m yardline_intelligence.pipeline
```
