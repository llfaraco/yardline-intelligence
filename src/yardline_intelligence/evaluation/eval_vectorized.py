"""Avaliação vetorizada rápida — v0.3.

Pré-calcula médias móveis usando window functions do pandas (groupby + shift +
rolling), evitando o processamento linha-a-linha que tornava o pipeline lento.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from yardline_intelligence.temporal_feature_store.store import moving_average_baseline


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


def precompute_sma(df: pd.DataFrame, window: int = 3) -> pd.Series:
    """Pré-calcula SMA-3 para cada (sleeper_id, week) de forma vetorizada.

    Retorna Series indexada pelo mesmo índice do df com o valor do baseline.
    Usa apenas semanas anteriores (shift >= 1), sem data leakage.
    """
    df = df.sort_values(["sleeper_id", "week"]).copy()
    g = df.groupby("sleeper_id")
    sma = g["half_ppr_points"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return sma


def evaluate_baseline_vectorized(df: pd.DataFrame,
                                 validate_weeks: list[int],
                                 window: int = 3) -> dict:
    """Avalia baseline SMA de forma vetorizada."""
    df = df.sort_values(["sleeper_id", "week"]).copy()
    sma = precompute_sma(df, window=window)

    df_eval = df.copy()
    df_eval["sma"] = sma.values
    df_eval = df_eval[df_eval["week"].isin(validate_weeks)]
    df_eval = df_eval[df_eval["half_ppr_points"].notna()]
    df_eval = df_eval[df_eval["sma"].notna()]

    if df_eval.empty:
        return {"n": 0}

    p = df_eval["sma"].values
    t = df_eval["half_ppr_points"].values
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    # Spearman (correlação de ranking do baseline).
    spearman = _spearman(p.tolist(), t.tolist())

    # Start/Sit Win Rate.
    ss_right = ss_total = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j] or p[i] == p[j]:
                continue
            ss_total += 1
            if (p[i] > p[j]) == (t[i] > t[j]):
                ss_right += 1
    ss = (ss_right / ss_total) if ss_total else None

    return {
        "n": int(n), "mae": round(mae, 4), "rmse": round(rmse, 4),
        "bias": round(bias, 4), "spearman": round(spearman, 4)
        if spearman is not None else None,
        "start_sit_win_rate": round(ss, 4)
        if ss is not None else None,
    }


def evaluate_residual_vectorized(model, df: pd.DataFrame,
                                 validate_weeks: list[int],
                                 feature_cols: list[str],
                                 window: int = 3) -> dict:
    """Avalia modelo residual de forma vetorizada.

    O modelo já deve estar treinado. Aplica predição por linha (necessário
    porque o modelo precisa do baseline por jogador), mas pré-calcula o SMA.
    """
    df = df.sort_values(["sleeper_id", "week"]).copy()
    sma = precompute_sma(df, window=window)

    df_eval = df.copy()
    df_eval["sma"] = sma.values
    df_eval = df_eval[df_eval["week"].isin(validate_weeks)]
    df_eval = df_eval[df_eval["half_ppr_points"].notna()]
    df_eval = df_eval[df_eval["sma"].notna()]

    if df_eval.empty:
        return {"n": 0}

    preds = []
    targets = []
    base_preds = []
    weeks = []

    # Pré-calcula features para todas as linhas de uma vez (vetorizado).
    X = df_eval[feature_cols].fillna(0).values
    # Predição em lote: ridge usa coef_ + intercept_, outros usam model.predict.
    if hasattr(model, "model") and model.model is not None:
        adj = model.model.predict(X)
    elif hasattr(model, "coef_") and model.coef_ is not None:
        adj = model.intercept_ + X @ model.coef_
    else:
        adj = None

    if adj is not None:
        # Remove NaN.
        adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

    if adj is not None:
        for i, (_, r) in enumerate(df_eval.iterrows()):
            base = float(r["sma"])
            # Aplica shrinkage manualmente (o modelo já treinou com shrinkage).
            pred = base + float(adj[i])
            preds.append(pred)
            targets.append(float(r["half_ppr_points"]))
            base_preds.append(base)
            weeks.append(int(r["week"]))

    if not preds:
        return {"n": 0}

    p = np.array(preds, dtype=float)
    t = np.array(targets, dtype=float)
    b = np.array(base_preds, dtype=float)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    # Spearman.
    def _spearman(x, y):
        if len(x) < 3:
            return None
        xr = _rank(x); yr = _rank(y)
        nn = len(xr)
        mx = sum(xr) / nn; my = sum(yr) / nn
        num = sum((a - mx) * (b - my) for a, b in zip(xr, yr))
        den = (sum((a - mx) ** 2 for a in xr) * sum((b - my) ** 2 for b in yr)) ** 0.5
        if den == 0:
            return None
        return num / den

    def _rank(values):
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

    spearman = _spearman(preds, targets)

    # Start/Sit.
    ss_right = ss_total = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j] or p[i] == p[j]:
                continue
            ss_total += 1
            if (p[i] > p[j]) == (t[i] > t[j]):
                ss_right += 1
    ss = (ss_right / ss_total) if ss_total else None

    base_mae = float(np.mean(np.abs(b - t)))
    imp = (base_mae - mae) / base_mae * 100 if base_mae else None

    return {
        "n": int(n), "mae": round(mae, 4), "rmse": round(rmse, 4),
        "bias": round(bias, 4), "spearman": round(spearman, 4)
        if spearman is not None else None,
        "start_sit_win_rate": round(ss, 4) if ss is not None else None,
        "baseline_mae": round(base_mae, 4),
        "improvement_vs_baseline_pct": round(imp, 2) if imp is not None else None,
        "baseline_start_sit": round(ss, 4) if ss is not None else None,
    }
