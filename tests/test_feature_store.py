"""Testes da camada 2 — Temporal Feature Store (features sem data leakage)."""

from __future__ import annotations

import pandas as pd

from yardline_intelligence.temporal_feature_store.store import (
    FeatureStore, moving_average_baseline,
)


def _df():
    rows = []
    for pid in ["WR001", "WR002"]:
        for week in range(1, 9):
            rows.append({
                "season": 2026, "week": week, "sleeper_id": pid,
                "gsis_id": pid, "name": pid, "position": "WR",
                "team": "T1", "opponent": "T2",
                "half_ppr_points": round(5.0 + week * 0.5, 2),
                "targets": week, "catches": week, "air_yards": week * 7.0,
                "receiving_yards": week * 6.0, "epa_sum": round(week * 0.1, 3),
                "rushes": 0, "rush_yards": 0.0,
                "dropbacks": 0, "pass_attempts": 0, "completions": 0, "sacks": 0,
                "snap_share": round(0.3 + week * 0.02, 3), "snap_count": 40 + week * 2,
                "games_played": week,
            })
    return pd.DataFrame(rows)


def test_feature_store_features_for_no_leakage():
    df = _df()
    store = FeatureStore(2026, from_df=df)
    # Semana 5 deve enxergar apenas semanas 1-4.
    snap = store.features_for("WR001", 5)
    assert snap.target is not None
    # snap_share_prev deve ser o da semana 4 (0.3 + 4*0.02 = 0.38).
    assert abs(snap.features["snap_share_prev"] - 0.38) < 1e-9
    # targets_mean_3 = média das semanas 2,3,4 = (2+3+4)/3 = 3.
    assert abs(snap.features["targets_mean_3"] - 3.0) < 1e-9
    # Não deve enxergar a semana 5 nem futuras.
    assert snap.features["targets_prev"] == 4.0


def test_feature_store_empty_before_first_week():
    df = _df()
    store = FeatureStore(2026, from_df=df)
    snap = store.features_for("WR001", 1)
    assert snap.features["snap_share_prev"] is None
    assert snap.features["targets_mean_3"] is None
    assert snap.target is not None


def test_moving_average_baseline_none_when_no_history():
    df = _df()
    store = FeatureStore(2026, from_df=df)
    assert moving_average_baseline(store, "WR001", 1) is None


def test_moving_average_baseline_value():
    df = _df()
    store = FeatureStore(2026, from_df=df)
    # Semana 4: média das semanas 1-3 = (5.5 + 6.0 + 6.5)/3 = 6.0
    val = moving_average_baseline(store, "WR001", 4, window=3)
    assert abs(val - 6.0) < 1e-9
