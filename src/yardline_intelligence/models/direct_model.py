"""Modelo direto v0.2 — predição de pontos Half-PPR a partir de todas as features.

Em vez de compor volume × eficiência (que propaga erros de dois modelos), este
modelo prediz diretamente os pontos usando regressão (Ridge / Elastic Net /
LightGBM / XGBoost) sobre o conjunto completo de features temporais.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..temporal_feature_store.expanded_store import ExpandedFeatureStore
from ..temporal_feature_store.store import moving_average_baseline


class DirectPointsModel:
    """Prediz pontos Half-PPR diretamente de features temporais."""

    def __init__(self, kind: str = "ridge", alpha: float = 1.0,
                 feature_cols: Optional[list[str]] = None):
        self.kind = kind
        self.alpha = alpha
        self.feature_cols = feature_cols or [
            "snap_share_prev", "snap_share_mean_3", "snap_share_mean_5",
            "snap_share_trend", "targets_mean_3", "targets_mean_5",
            "targets_prev", "targets_trend", "games_played_prev_n", "weeks_active",
            "target_consistency",
        ]
        self.coef_ = None
        self.intercept_ = 0.0
        self.model = None
        self.fitted = False

    def fit(self, store: ExpandedFeatureStore, through_week: int,
            position: str = "WR") -> "DirectPointsModel":
        df = store.training_matrix(through_week, positions=(position,))
        if df.empty:
            self.fitted = False
            return self

        X = []
        y = []
        for _, r in df.iterrows():
            feats = store.features_for(r["sleeper_id"], int(r["week"]))
            X.append(_feature_vector(feats, self.feature_cols))
            y.append(float(r["half_ppr_points"]))
        X = np.array(X)
        y = np.array(y)

        if self.kind == "ridge":
            Xc = X - X.mean(axis=0, keepdims=True)
            yc = y - y.mean()
            A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
            b = Xc.T @ yc
            try:
                coef = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(A, b, rcond=None)[0]
            self.coef_ = coef
            self.intercept_ = float(y.mean() - (Xc.mean(axis=0) @ coef))
        elif self.kind == "elastic_net":
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet(alpha=self.alpha, l1_ratio=0.5, max_iter=5000, random_state=42)
            self.model.fit(X, y)
        elif self.kind == "lightgbm":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                               max_depth=6, random_state=42, verbosity=-1)
                self.model.fit(X, y)
            except Exception:
                self.kind = "ridge"
                return self.fit(store, through_week, position)
        elif self.kind == "xgboost":
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                              max_depth=6, random_state=42, verbosity=0)
                self.model.fit(X, y)
            except Exception:
                self.kind = "ridge"
                return self.fit(store, through_week, position)

        self.fitted = True
        return self

    def predict(self, feats: dict) -> Optional[float]:
        if not self.fitted:
            return None
        if self.kind == "ridge":
            if self.coef_ is None:
                return None
            val = float(self.intercept_ + _feature_vector(feats, self.feature_cols) @ self.coef_)
            return float(max(0.0, val))
        if self.model is not None:
            return float(max(0.0, self.model.predict([_feature_vector(feats, self.feature_cols)])[0]))
        return None


def _feature_vector(feats: dict, feature_cols: list[str]) -> np.ndarray:
    vals = []
    for c in feature_cols:
        v = feats.get(c)
        try:
            v = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            v = 0.0
        vals.append(v)
    return np.array(vals, dtype=float)


def _spearman(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    xr = _rank(x)
    yr = _rank(y)
    n = len(xr)
    mx = sum(xr) / n
    my = sum(yr) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xr, yr))
    den = (sum((a - mx) ** 2 for a in xr) * sum((b - my) ** 2 for b in yr)) ** 0.5
    if den == 0:
        return None
    return num / den


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def evaluate_model(store: ExpandedFeatureStore, model, validate_weeks: list[int],
                   position: str = "WR") -> dict:
    """Avalia um modelo de pontos diretos em semanas de validação.

    Retorna dicionário com MAE, RMSE, Bias, Spearman, Start/Sit Win Rate
    e comparação com baseline SMA-3.
    """
    import numpy as np

    errors = []
    base_errors = []
    preds = []
    targets = []
    base_preds = []

    for week in validate_weeks:
        players = store.players(week)
        for sid in players:
            feats = store.features_for(sid, week)
            target = feats.get("half_ppr_points")
            if target is None:
                continue
            pred = model.predict(feats)
            if pred is None:
                continue
            base = moving_average_baseline(store, sid, week, window=3)
            errors.append(abs(pred - target))
            preds.append(pred)
            targets.append(float(target))
            if base is not None:
                base_errors.append(abs(base - target))
                base_preds.append(base)

    n = len(preds)
    if n == 0:
        return {"n": 0}

    p = np.array(preds, dtype=float)
    t = np.array(targets, dtype=float)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    spearman = _spearman(preds, targets)

    ss_right = 0
    ss_total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j] or p[i] == p[j]:
                continue
            ss_total += 1
            if (p[i] > p[j]) == (t[i] > t[j]):
                ss_right += 1
    ss_win_rate = (ss_right / ss_total) if ss_total else None

    out = {
        "n": n, "mae": round(mae, 4), "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "spearman": round(spearman, 4) if spearman is not None else None,
        "start_sit_win_rate": round(ss_win_rate, 4) if ss_win_rate is not None else None,
        "model_mean_pred": round(float(p.mean()), 3),
        "model_std_pred": round(float(p.std()), 3),
        "target_mean": round(float(t.mean()), 3),
        "target_std": round(float(t.std()), 3),
    }

    if base_preds and len(base_preds) == n:
        b = np.array(base_preds, dtype=float)
        base_mae = float(np.mean(np.abs(b - t)))
        base_bias = float(np.mean(b - t))
        base_spear = _spearman(base_preds, targets)

        b_ss_right = 0
        b_ss_total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if t[i] == t[j] or b[i] == b[j]:
                    continue
                b_ss_total += 1
                if (b[i] > b[j]) == (t[i] > t[j]):
                    b_ss_right += 1
        base_ss = (b_ss_right / b_ss_total) if b_ss_total else None

        out["baseline_mae"] = round(base_mae, 4)
        out["baseline_bias"] = round(base_bias, 4)
        out["baseline_spearman"] = round(base_spear, 4) if base_spear is not None else None
        out["baseline_start_sit"] = round(base_ss, 4) if base_ss is not None else None
        out["improvement_vs_baseline_pct"] = round(
            (base_mae - mae) / base_mae * 100, 2) if base_mae else None
        out["baseline_mean_pred"] = round(float(b.mean()), 3)

    return out
