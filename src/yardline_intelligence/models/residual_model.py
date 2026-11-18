"""Modelo híbrido residual — v0.3.

Hipótese: Projeção final = Baseline histórico + Ajuste contextual.
O ML aprende a prever o ERRO do baseline (resíduo), em vez de reconstruir
toda a pontuação Fantasy. Isso reduz variância e evita alterações excessivas
em projeções estáveis.

Modelos comparados:
- SMA-3, SMA-5, EMA (baselines puros)
- Ridge residual, Elastic Net residual, LightGBM residual
- Ensemble ponderado
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..temporal_feature_store.store import moving_average_baseline
from ..temporal_feature_store.expanded_store import ExpandedFeatureStore


@dataclass
class ResidualResult:
    name: str
    position: str
    season: int
    train_weeks: int
    validate_weeks: int
    n_rows: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    bias: Optional[float] = None
    spearman: Optional[float] = None
    start_sit_win_rate: Optional[float] = None
    baseline_mae: Optional[float] = None
    improvement_vs_baseline_pct: Optional[float] = None
    ablation_group: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "position": self.position, "season": self.season,
            "train_weeks": self.train_weeks, "validate_weeks": self.validate_weeks,
            "n_rows": self.n_rows, "mae": self.mae, "rmse": self.rmse,
            "bias": self.bias, "spearman": self.spearman,
            "start_sit_win_rate": self.start_sit_win_rate,
            "baseline_mae": self.baseline_mae,
            "improvement_vs_baseline_pct": self.improvement_vs_baseline_pct,
            "ablation_group": self.ablation_group, "notes": self.notes,
        }


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


class ResidualModel:
    """Modelo que prediz o resíduo do baseline: pontos = baseline + ajuste.

    O ajuste é regularizado para evitar alterações excessivas em projeções
    estáveis (shrinkage do resíduo).
    """

    def __init__(self, kind: str = "ridge", alpha: float = 1.0,
                 shrinkage: float = 0.5,
                 feature_cols: Optional[list[str]] = None):
        self.kind = kind
        self.alpha = alpha
        self.shrinkage = shrinkage  # 0 = ajuste total, 1 = nenhum ajuste (só baseline)
        self.feature_cols = feature_cols or [
            "snap_share_prev", "snap_share_mean_3", "snap_share_mean_5",
            "targets_mean_3", "targets_mean_5", "targets_prev", "targets_trend",
            "games_played_prev_n", "weeks_active", "target_consistency",
        ]
        self.coef_ = None
        self.intercept_ = 0.0
        self.model = None
        self.fitted = False

    def fit(self, store: ExpandedFeatureStore, through_week: int,
            position: str = "WR") -> "ResidualModel":
        df = store.training_matrix(through_week, positions=(position,))
        if df.empty:
            self.fitted = False
            return self

        # Constroi matriz com baseline e alvo.
        X = []
        y_resid = []
        baselines = []
        for _, r in df.iterrows():
            sid = r["sleeper_id"]
            week = int(r["week"])
            feats = store.features_for(sid, week)
            base = moving_average_baseline(store, sid, week, window=3)
            if base is None:
                continue
            X.append(_feature_vector(feats, self.feature_cols))
            y_resid.append(float(r["half_ppr_points"]) - base)
            baselines.append(base)

        if not X:
            self.fitted = False
            return self

        X = np.array(X)
        y_resid = np.array(y_resid)

        # Regulariza o resíduo: encolhe em direção a zero.
        y_resid = y_resid * (1.0 - self.shrinkage)

        if self.kind == "ridge":
            Xc = X - X.mean(axis=0, keepdims=True)
            yc = y_resid - y_resid.mean()
            A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
            b = Xc.T @ yc
            try:
                coef = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(A, b, rcond=None)[0]
            self.coef_ = coef
            self.intercept_ = float(y_resid.mean() - (Xc.mean(axis=0) @ coef))
        elif self.kind == "elastic_net":
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet(alpha=self.alpha, l1_ratio=0.5,
                                    max_iter=5000, random_state=42)
            self.model.fit(X, y_resid)
        elif self.kind == "lightgbm":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                               max_depth=6, random_state=42,
                                               verbosity=-1)
                self.model.fit(X, y_resid)
            except Exception:
                self.kind = "ridge"
                return self.fit(store, through_week, position)

        self.fitted = True
        return self

    def predict_with_baseline(self, feats: dict,
                              store: Optional[ExpandedFeatureStore] = None,
                              sleeper_id: Optional[str] = None,
                              week: Optional[int] = None) -> Optional[float]:
        """Retorna baseline + ajuste residual (com shrinkage aplicado)."""
        if not self.fitted:
            return None
        if store is None or sleeper_id is None or week is None:
            return None
        base = moving_average_baseline(store, sleeper_id, week, window=3)
        if base is None:
            return None
        adj = self.predict(feats)
        if adj is None:
            return base
        return float(base + adj)

    def predict(self, feats: dict) -> Optional[float]:
        """Retorna apenas o ajuste residual (já com shrinkage)."""
        if not self.fitted:
            return None
        if self.kind == "ridge":
            if self.coef_ is None:
                return None
            val = float(self.intercept_ + _feature_vector(feats, self.feature_cols) @ self.coef_)
            return val
        if self.model is not None:
            return float(self.model.predict([_feature_vector(feats, self.feature_cols)])[0])
        return None


def evaluate_residual(store: ExpandedFeatureStore, model: ResidualModel,
                      validate_weeks: list[int],
                      position: str = "WR",
                      baseline_window: int = 3) -> ResidualResult:
    """Avalia modelo residual em semanas de validação (walk-forward)."""
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
            pred = model.predict_with_baseline(feats, store, sid, week)
            if pred is None:
                continue
            base = moving_average_baseline(store, sid, week, window=baseline_window)
            errors.append(abs(pred - target))
            base_errors.append(abs(base - target))
            preds.append(pred)
            targets.append(float(target))
            base_preds.append(base)

    n = len(preds)
    if n == 0:
        return ResidualResult(
            name=getattr(model, "kind", "residual"), position=position,
            season=store.season, train_weeks=max(validate_weeks) - 2
            if validate_weeks else 0, validate_weeks=len(validate_weeks),
            n_rows=0, ablation_group=getattr(model, "ablation_group", None),
            notes="sem predições válidas",
        )

    p = np.array(preds, dtype=float)
    t = np.array(targets, dtype=float)
    b = np.array(base_preds, dtype=float)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    spearman = _spearman(preds, targets)

    # Start/Sit Win Rate.
    ss_right = ss_total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j] or p[i] == p[j]:
                continue
            ss_total += 1
            if (p[i] > p[j]) == (t[i] > t[j]):
                ss_right += 1
    ss_win_rate = (ss_right / ss_total) if ss_total else None

    base_mae = float(np.mean(np.abs(b - t)))
    imp = (base_mae - mae) / base_mae * 100 if base_mae else None

    return ResidualResult(
        name=getattr(model, "kind", "residual"), position=position,
        season=store.season, train_weeks=max(validate_weeks) - 2
        if validate_weeks else 0, validate_weeks=len(validate_weeks),
        n_rows=n, mae=round(mae, 4), rmse=round(rmse, 4),
        bias=round(bias, 4), spearman=round(spearman, 4)
        if spearman is not None else None,
        start_sit_win_rate=round(ss_win_rate, 4)
        if ss_win_rate is not None else None,
        baseline_mae=round(base_mae, 4),
        improvement_vs_baseline_pct=round(imp, 2) if imp is not None else None,
        ablation_group=getattr(model, "ablation_group", None),
    )


def baseline_sma(store: ExpandedFeatureStore, validate_weeks: list[int],
                 window: int = 3, position: str = "WR") -> ResidualResult:
    """Calcula métricas do baseline SMA puro para comparação justa."""
    errors = []
    preds = []
    targets = []
    for week in validate_weeks:
        players = store.players(week)
        for sid in players:
            feats = store.features_for(sid, week)
            target = feats.get("half_ppr_points")
            if target is None:
                continue
            base = moving_average_baseline(store, sid, week, window=window)
            if base is None:
                continue
            errors.append(abs(base - target))
            preds.append(base)
            targets.append(float(target))

    n = len(preds)
    if n == 0:
        return ResidualResult("sma", position, store.season, 0, len(validate_weeks),
                              0, ablation_group="A", notes="sem dados")

    p = np.array(preds, dtype=float)
    t = np.array(targets, dtype=float)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    spearman = _spearman(preds, targets)

    ss_right = ss_total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j] or p[i] == p[j]:
                continue
            ss_total += 1
            if (p[i] > p[j]) == (t[i] > t[j]):
                ss_right += 1
    ss_win_rate = (ss_right / ss_total) if ss_total else None

    return ResidualResult(
        name=f"sma_{window}", position=position, season=store.season,
        train_weeks=0, validate_weeks=len(validate_weeks), n_rows=n,
        mae=round(mae, 4), rmse=round(rmse, 4), bias=round(bias, 4),
        spearman=round(spearman, 4) if spearman is not None else None,
        start_sit_win_rate=round(ss_win_rate, 4) if ss_win_rate is not None else None,
        baseline_mae=round(mae, 4), improvement_vs_baseline_pct=0.0,
        ablation_group="A", notes=f"Média móvel simples de {window} semanas (baseline)",
    )
