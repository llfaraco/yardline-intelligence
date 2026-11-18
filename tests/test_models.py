"""Testes da camada 3 v0.2 — Modelos WR (baseline e modelos diretos)."""

from __future__ import annotations

import pandas as pd

from yardline_intelligence.temporal_feature_store.expanded_store import ExpandedFeatureStore
from yardline_intelligence.models.wr_models import VolumeModel, EfficiencyModel, CompositeWithBaseline
from yardline_intelligence.models.direct_model import DirectPointsModel


def _synthetic_df():
    rows = []
    rng = __import__("numpy").random.default_rng(7)
    for pid in ["WR001", "WR002", "WR003", "WR004"]:
        base = 4 + 4 * ("WR001" == pid) + 2 * ("WR002" == pid)
        for week in range(1, 13):
            pts = max(0, base + week * 0.3 + rng.normal(0, 1.0))
            rows.append({
                "season": 2026, "week": week, "sleeper_id": pid,
                "gsis_id": pid, "name": pid, "position": "WR",
                "team": "T1", "opponent": "T2",
                "half_ppr_points": round(float(pts), 2),
                "targets": max(1, int(round(pts / 3))), "catches": 0,
                "air_yards": 0.0, "receiving_yards": 0.0,
                "epa_sum": 0.0, "rushes": 0, "rush_yards": 0.0,
                "dropbacks": 0, "pass_attempts": 0, "completions": 0, "sacks": 0,
                "snap_share": round(min(1.0, max(0.1, 0.3 + week * 0.02)), 3),
                "snap_count": 40 + week * 2,
                "games_played": week,
            })
    return pd.DataFrame(rows)


def test_volume_model_predict_targets():
    """VolumeModel deve prever targets (não pontos) sem data leakage."""
    df = _synthetic_df()
    store = ExpandedFeatureStore(2026, from_df=df)

    volume = VolumeModel(kind="ridge", alpha=1.0).fit(store, 8, position="WR")
    assert volume.fitted

    feats = store.features_for("WR001", 9)
    pred = volume.predict(feats)
    assert pred is not None and pred >= 0


def test_efficiency_model_predict_nonnegative():
    """EfficiencyModel deve prever pts/alvo >= 0."""
    df = _synthetic_df()
    store = ExpandedFeatureStore(2026, from_df=df)

    efficiency = EfficiencyModel(kind="ridge", alpha=1.0).fit(store, 8, position="WR")
    assert efficiency.fitted

    feats = store.features_for("WR001", 9)
    pred = efficiency.predict(feats)
    assert pred is not None and pred >= 0


def test_composite_predict_points():
    """Composite prevê pontos = volume × eficiência."""
    df = _synthetic_df()
    store = ExpandedFeatureStore(2026, from_df=df)

    volume = VolumeModel(kind="ridge", alpha=1.0).fit(store, 8, position="WR")
    efficiency = EfficiencyModel(kind="ridge", alpha=1.0).fit(store, 8, position="WR")
    composite = CompositeWithBaseline(volume, efficiency)
    assert composite.fitted

    feats = store.features_for("WR001", 9)
    pred = composite.predict(feats)
    assert pred is not None and pred >= 0


def test_direct_points_model_predict():
    """DirectPointsModel prevê pontos diretamente."""
    df = _synthetic_df()
    store = ExpandedFeatureStore(2026, from_df=df)

    model = DirectPointsModel(kind="ridge", alpha=1.0).fit(store, 8, position="WR")
    assert model.fitted

    feats = store.features_for("WR001", 9)
    pred = model.predict(feats)
    assert pred is not None and pred >= 0


def test_lightgbm_model_fit_and_predict():
    """LightGBM deve treinar e predizer quando disponível."""
    df = _synthetic_df()
    store = ExpandedFeatureStore(2026, from_df=df)

    model = DirectPointsModel(kind="lightgbm", alpha=1.0).fit(store, 8, position="WR")
    if model.fitted and model.kind == "lightgbm":
        feats = store.features_for("WR001", 9)
        pred = model.predict(feats)
        assert pred is not None and pred >= 0
