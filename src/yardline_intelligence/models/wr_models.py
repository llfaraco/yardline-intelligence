"""Camada 3 v0.2 — Modelos especializados e experimentação de ablação.

Diagnóstico do v0.1: o VolumeModel estava prevendo half_ppr_points em vez de
targets (alvos), o que quebra a composição volume × eficiência.

Nesta versão:
- VolumeModel prevê TARGETS (não pontos).
- EfficiencyModel prevê PONTOS POR ALVO.
- Composição: pontos = targets_pred × pts_por_target_pred.
- Comparação de múltiplas famílias: média móvel simples, Ridge, Elastic Net,
  LightGBM, XGBoost e ensemble.
- Accuracy Lab com MAE, RMSE, Bias, Spearman, Start/Sit Win Rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..temporal_feature_store.expanded_store import ExpandedFeatureStore
from ..temporal_feature_store.store import moving_average_baseline


# =====================================================================
# Resultado de experimento
# =====================================================================

@dataclass
class ModelResult:
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
    decision_regret: Optional[float] = None
    coverage: Optional[float] = None
    baseline_mae: Optional[float] = None
    improvement_vs_baseline_pct: Optional[float] = None
    baseline_start_sit: Optional[float] = None
    ablation_group: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "position": self.position, "season": self.season,
            "train_weeks": self.train_weeks, "validate_weeks": self.validate_weeks,
            "n_rows": self.n_rows, "mae": self.mae, "rmse": self.rmse,
            "bias": self.bias, "spearman": self.spearman,
            "start_sit_win_rate": self.start_sit_win_rate,
            "decision_regret": self.decision_regret, "coverage": self.coverage,
            "baseline_mae": self.baseline_mae,
            "improvement_vs_baseline_pct": self.improvement_vs_baseline_pct,
            "baseline_start_sit": self.baseline_start_sit,
            "ablation_group": self.ablation_group, "notes": self.notes,
        }


# =====================================================================
# Métricas do Accuracy Lab
# =====================================================================

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


def accuracy_metrics(preds: list[float], targets: list[float],
                     base_preds: Optional[list[float]] = None) -> dict:
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

    # Start/Sit Win Rate: pares de mesma semana ordenados corretamente.
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
        "decision_regret": None,
        "coverage": None,
    }

    if base_preds is not None and len(base_preds) == n:
        b = np.array(base_preds, dtype=float)
        base_mae = float(np.mean(np.abs(b - t)))
        out["baseline_mae"] = round(base_mae, 4)
        out["improvement_vs_baseline_pct"] = round(
            (base_mae - mae) / base_mae * 100, 2) if base_mae else None

        b_ss_right = 0
        b_ss_total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if t[i] == t[j] or b[i] == b[j]:
                    continue
                b_ss_total += 1
                if (b[i] > b[j]) == (t[i] > t[j]):
                    b_ss_right += 1
        out["baseline_start_sit"] = round(b_ss_right / b_ss_total, 4) if b_ss_total else None

    return out


# =====================================================================
# Vetor de features
# =====================================================================

def _feature_vector(snapshot_features: dict, feature_cols: list[str]) -> np.ndarray:
    vals = []
    for c in feature_cols:
        v = snapshot_features.get(c)
        try:
            v = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            v = 0.0
        vals.append(v)
    return np.array(vals, dtype=float)


# =====================================================================
# Modelo de volume (prevê targets)
# =====================================================================

class VolumeModel:
    """Prevê targets (alvos) de um WR para uma semana."""

    def __init__(self, kind: str = "ridge", alpha: float = 1.0,
                 feature_cols: Optional[list[str]] = None):
        self.kind = kind
        self.alpha = alpha
        self.feature_cols = feature_cols or [
            "snap_share_prev", "snap_share_mean_3", "snap_share_mean_5",
            "snap_share_trend", "targets_mean_3", "targets_mean_5",
            "targets_prev", "targets_trend", "games_played_prev_n", "weeks_active",
        ]
        self.coef_ = None
        self.intercept_ = 0.0
        self.model = None
        self.fitted = False

    def fit(self, store: ExpandedFeatureStore, through_week: int,
            position: str = "WR") -> "VolumeModel":
        df = store.training_matrix(through_week, positions=(position,))
        if df.empty:
            self.fitted = False
            return self
        df = df[df["targets"].notna()]
        if df.empty:
            self.fitted = False
            return self

        if self.kind in ("sma", "ema"):
            self.fitted = True
            return self

        X = []
        y = []
        for _, r in df.iterrows():
            feats = store.features_for(r["sleeper_id"], int(r["week"]))
            X.append(_feature_vector(feats, self.feature_cols))
            y.append(float(r["targets"]))
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
                self.model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                               max_depth=6, random_state=42, verbosity=-1)
                self.model.fit(X, y)
            except Exception:
                self.kind = "ridge"
                return self.fit(store, through_week, position)
        elif self.kind == "xgboost":
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
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
        if self.kind in ("sma", "ema"):
            return None  # tratado pelo baseline externo
        if self.kind == "ridge":
            if self.coef_ is None:
                return None
            val = float(self.intercept_ + _feature_vector(feats, self.feature_cols) @ self.coef_)
            return float(max(0.0, val))  # targets não podem ser negativos
        if self.model is not None:
            return float(max(0.0, self.model.predict([_feature_vector(feats, self.feature_cols)])[0]))
        return None


# =====================================================================
# Modelo de eficiência (prevê pts por alvo)
# =====================================================================

class EfficiencyModel:
    """Prevê pontos Half-PPR por alvo de um WR."""

    def __init__(self, kind: str = "ridge", alpha: float = 1.0,
                 feature_cols: Optional[list[str]] = None):
        self.kind = kind
        self.alpha = alpha
        self.feature_cols = feature_cols or [
            "epa_per_target_mean_3", "epa_per_target_mean_5",
            "air_yards_per_target_mean_3", "air_yards_per_target_mean_5",
            "snap_share_mean_3", "snap_share_trend", "targets_mean_3",
            "games_played_prev_n", "weeks_active",
        ]
        self.coef_ = None
        self.intercept_ = 0.0
        self.model = None
        self.fitted = False

    def fit(self, store: ExpandedFeatureStore, through_week: int,
            position: str = "WR") -> "EfficiencyModel":
        df = store.training_matrix(through_week, positions=(position,))
        if df.empty:
            self.fitted = False
            return self

        X = []
        y = []
        for _, r in df.iterrows():
            feats = store.features_for(r["sleeper_id"], int(r["week"]))
            tgt = feats.get("targets_prev")
            try:
                tgt = float(tgt) if tgt is not None else 0.0
            except (TypeError, ValueError):
                tgt = 0.0
            if tgt <= 0:
                continue
            X.append(_feature_vector(feats, self.feature_cols))
            pts = float(r["half_ppr_points"])
            y.append(pts / max(tgt, 1.0))
        if not X:
            self.fitted = False
            return self

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
        elif self.kind in ("lightgbm", "xgboost"):
            try:
                if self.kind == "lightgbm":
                    import lightgbm as lgb
                    self.model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                                   max_depth=6, random_state=42, verbosity=-1)
                else:
                    import xgboost as xgb
                    self.model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
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
            return float(max(0.0, val))  # eficiência não pode ser negativa
        if self.model is not None:
            return float(max(0.0, self.model.predict([_feature_vector(feats, self.feature_cols)])[0]))
        return None


# =====================================================================
# Modelo composto (volume × eficiência)
# =====================================================================

class CompositeWRModel:
    """Projeção Half-PPR = targets_pred × pts_por_target_pred."""

    def __init__(self, volume: VolumeModel, efficiency: EfficiencyModel):
        self.volume = volume
        self.efficiency = efficiency
        self.fitted = volume.fitted and efficiency.fitted

    def predict(self, feats: dict) -> Optional[float]:
        if not self.fitted:
            return None
        vol = self.volume.predict(feats)
        eff = self.efficiency.predict(feats)
        if vol is None or eff is None:
            return None
        return float(max(0.0, vol * eff))


# =====================================================================
# Avaliador walk-forward com Accuracy Lab
# =====================================================================

def evaluate_model(store: ExpandedFeatureStore, model, validate_weeks: list[int],
                   position: str = "WR",
                   baseline_kind: str = "sma") -> ModelResult:
    """Avalia um modelo em semanas de validação (walk-forward)."""
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

    m = accuracy_metrics(preds, targets, base_preds if base_preds else None)

    return ModelResult(
        name=getattr(model, "kind", "composite"),
        position=position, season=store.season,
        train_weeks=max(validate_weeks) - 2 if validate_weeks else 0,
        validate_weeks=len(validate_weeks),
        n_rows=len(preds),
        mae=m.get("mae"), rmse=m.get("rmse"), bias=m.get("bias"),
        spearman=m.get("spearman"),
        start_sit_win_rate=m.get("start_sit_win_rate"),
        decision_regret=m.get("decision_regret"),
        coverage=m.get("coverage"),
        baseline_mae=m.get("baseline_mae"),
        improvement_vs_baseline_pct=m.get("improvement_vs_baseline_pct"),
        baseline_start_sit=m.get("baseline_start_sit"),
    )


class BaselineWrapper:
    """Média móvel simples, exposto como modelo para avaliação uniforme."""
    kind = "sma"
    fitted = True

    def predict(self, feats: dict, store: Optional[ExpandedFeatureStore] = None,
                sleeper_id: Optional[str] = None, week: Optional[int] = None) -> Optional[float]:
        if store is None or sleeper_id is None or week is None:
            return None
        return moving_average_baseline(store, sleeper_id, week, window=3)


class CompositeWithBaseline:
    """Composite que passa o store para o VolumeModel (precisa do store para SMA/EMA)."""
    def __init__(self, volume: VolumeModel, efficiency: EfficiencyModel):
        self.volume = volume
        self.efficiency = efficiency
        self.fitted = volume.fitted and efficiency.fitted
        self.kind = "composite"

    def predict(self, feats: dict, store: Optional[ExpandedFeatureStore] = None,
                sleeper_id: Optional[str] = None, week: Optional[int] = None) -> Optional[float]:
        if not self.fitted:
            return None
        vol = self.volume.predict(feats)
        eff = self.efficiency.predict(feats)
        if vol is None or eff is None:
            return None
        return float(max(0.0, vol * eff))


class EnsembleWrapper:
    """Ensemble simples: média do modelo composite e do baseline."""
    kind = "ensemble"
    fitted = True

    def __init__(self, composite: CompositeWithBaseline,
                 baseline: BaselineWrapper):
        self.composite = composite
        self.baseline = baseline

    def predict(self, feats: dict, store: Optional[ExpandedFeatureStore] = None,
                sleeper_id: Optional[str] = None, week: Optional[int] = None) -> Optional[float]:
        p1 = self.composite.predict(feats, store, sleeper_id, week)
        p2 = self.baseline.predict(feats, store, sleeper_id, week)
        if p1 is None and p2 is None:
            return None
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        return float((p1 + p2) / 2)
