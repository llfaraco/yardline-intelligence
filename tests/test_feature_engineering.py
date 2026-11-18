"""Regression tests for season-scoped temporal feature engineering."""

import pandas as pd

from yardline_intelligence.feature_engineering.temporal import add_temporal_features


def test_rolling_features_do_not_cross_season_boundaries():
    df = pd.DataFrame(
        [
            {"season": 2024, "week": 1, "sleeper_id": "WR1", "targets": 8},
            {"season": 2024, "week": 2, "sleeper_id": "WR1", "targets": 10},
            {"season": 2025, "week": 1, "sleeper_id": "WR1", "targets": 2},
            {"season": 2025, "week": 2, "sleeper_id": "WR1", "targets": 4},
        ]
    )

    result = add_temporal_features(df)

    week_1_2025 = result[(result["season"] == 2025) & (result["week"] == 1)].iloc[0]
    week_2_2025 = result[(result["season"] == 2025) & (result["week"] == 2)].iloc[0]

    assert pd.isna(week_1_2025["targets_prev"])
    assert week_2_2025["targets_prev"] == 2
    assert week_2_2025["targets_mean_3"] == 2
    assert week_1_2025["weeks_active"] == 0
