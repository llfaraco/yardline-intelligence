"""Regression tests for contextual feature propagation into model snapshots."""

import pandas as pd

from yardline_intelligence.temporal_feature_store.expanded_store import ExpandedFeatureStore


def test_contextual_columns_are_exposed_by_features_for():
    df = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "sleeper_id": "WR1",
                "position": "WR",
                "team": "AAA",
                "opponent": "BBB",
                "half_ppr_points": 8.0,
                "targets": 5,
                "snap_share": 0.75,
                "games_played": 1,
                "jogos_perdidos_ultimas_3": 0,
                "qb_epa_por_dropback_ultimas_3": 0.12,
                "gs_vegas_total_linha": 44.5,
            },
            {
                "season": 2025,
                "week": 2,
                "sleeper_id": "WR1",
                "position": "WR",
                "team": "AAA",
                "opponent": "CCC",
                "half_ppr_points": 12.0,
                "targets": 7,
                "snap_share": 0.82,
                "games_played": 1,
                "jogos_perdidos_ultimas_3": 1,
                "qb_epa_por_dropback_ultimas_3": 0.24,
                "gs_vegas_total_linha": 48.0,
            },
        ]
    )
    store = ExpandedFeatureStore(2025, from_df=df)

    features = store.features_for("WR1", 2)

    assert features["jogos_perdidos_ultimas_3"] == 1
    assert features["qb_epa_por_dropback_ultimas_3"] == 0.24
    assert features["gs_vegas_total_linha"] == 48.0
