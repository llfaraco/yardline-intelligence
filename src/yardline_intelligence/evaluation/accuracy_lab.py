"""Accuracy Lab — v0.3.

Avaliação científica com:
- MAE, RMSE, Bias, Spearman, Start/Sit Win Rate, Decision Regret.
- Métricas por semana, faixa de projeção, titulares/reservas,
  jogadores retornando de lesão, mudanças de QB.
- Intervalos de confiança via bootstrap agrupado por semana (cluster bootstrap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AccuracyResult:
    name: str
    position: str
    season: int
    n_rows: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    bias: Optional[float] = None
    spearman: Optional[float] = None
    start_sit_win_rate: Optional[float] = None
    decision_regret: Optional[float] = None
    coverage_p10_p90: Optional[float] = None
    crps: Optional[float] = None
    ci_mae_lo: Optional[float] = None
    ci_mae_hi: Optional[float] = None
    ci_rmse_lo: Optional[float] = None
    ci_rmse_hi: Optional[float] = None
    ablation_group: Optional[str] = None
    by_week: Optional[dict] = None
    by_projection_band: Optional[dict] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "position": self.position, "season": self.season,
            "n_rows": self.n_rows, "mae": self.mae, "rmse": self.rmse,
            "bias": self.bias, "spearman": self.spearman,
            "start_sit_win_rate": self.start_sit_win_rate,
            "decision_regret": self.decision_regret,
            "coverage_p10_p90": self.coverage_p10_p90,
            "crps": self.crps,
            "ci_mae_lo": self.ci_mae_lo, "ci_mae_hi": self.ci_mae_hi,
            "ci_rmse_lo": self.ci_rmse_lo, "ci_rmse_hi": self.ci_rmse_hi,
            "ablation_group": self.ablation_group, "notes": self.notes,
            "by_week": self.by_week, "by_projection_band": self.by_projection_band,
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


def core_metrics(preds: list[float], targets: list[float],
                 base_preds: Optional[list[float]] = None,
                 p10: Optional[list[float]] = None,
                 p90: Optional[list[float]] = None) -> dict:
    """Calcula métricas centrais do Accuracy Lab."""
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

    out = {
        "n": n, "mae": round(mae, 4), "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "spearman": round(spearman, 4) if spearman is not None else None,
        "start_sit_win_rate": round(ss_win_rate, 4)
        if ss_win_rate is not None else None,
        "decision_regret": None,
    }

    if base_preds is not None and len(base_preds) == n:
        b = np.array(base_preds, dtype=float)
        base_mae = float(np.mean(np.abs(b - t)))
        out["baseline_mae"] = round(base_mae, 4)
        out["improvement_vs_baseline_pct"] = round(
            (base_mae - mae) / base_mae * 100, 2) if base_mae else None

        # Decision Regret: fração de vezes em que o modelo escolhe um jogador
        # diferente do baseline e o real é pior.
        regret_count = 0
        regret_total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if t[i] == t[j]:
                    continue
                # Modelo escolhe i, baseline escolhe j (ou vice-versa).
                model_picks_i = p[i] > p[j]
                base_picks_i = b[i] > b[j]
                if model_picks_i != base_picks_i:
                    regret_total += 1
                    # Real é pior para a escolha do modelo.
                    if model_picks_i and t[i] < t[j]:
                        regret_count += 1
                    if not model_picks_i and t[j] < t[i]:
                        regret_count += 1
        out["decision_regret"] = round(regret_count / regret_total, 4) \
            if regret_total else None

    # Cobertura dos intervalos P10-P90.
    if p10 is not None and p90 is not None and len(p10) == n:
        lo = np.array(p10, dtype=float)
        hi = np.array(p90, dtype=float)
        covered = ((t >= lo) & (t <= hi)).mean()
        out["coverage_p10_p90"] = round(float(covered), 4)

        # CRPS simplificado (usando intervalo P10-P90 como aproximação).
        # CRPS ≈ E|X - y| - 0.5 * E|X - X'| ; aproximado por largura do intervalo.
        width = hi - lo
        out["crps"] = round(float(width.mean()), 4)

    return out


def bootstrap_ci(preds: list[float], targets: list[float],
                 weeks: list[int],
                 metric: str = "mae",
                 n_bootstrap: int = 1000,
                 alpha: float = 0.05,
                 seed: int = 42) -> tuple[Optional[float], Optional[float]]:
    """Intervalo de confiança via bootstrap agrupado por semana (cluster bootstrap).

    Reamostra semanas (não observações individuais) para preservar
    correlacionamento dentro de cada semana.
    """
    if len(preds) < 10 or len(weeks) < 3:
        return None, None

    rng = np.random.default_rng(seed)
    unique_weeks = sorted(set(weeks))
    n_weeks = len(unique_weeks)

    # Mapeia cada observação para sua semana.
    week_to_idx = {w: [] for w in unique_weeks}
    for i, w in enumerate(weeks):
        week_to_idx[w].append(i)

    boot_stats = []
    for _ in range(n_bootstrap):
        # Reamostra semanas com reposição.
        sampled_weeks = rng.choice(unique_weeks, size=n_weeks, replace=True)
        idx = []
        for w in sampled_weeks:
            idx.extend(week_to_idx[w])
        if not idx:
            continue
        p_b = np.array([preds[i] for i in idx], dtype=float)
        t_b = np.array([targets[i] for i in idx], dtype=float)
        if metric == "mae":
            val = float(np.mean(np.abs(p_b - t_b)))
        elif metric == "rmse":
            val = float(np.sqrt(np.mean((p_b - t_b) ** 2)))
        else:
            val = float(np.mean(p_b - t_b))
        boot_stats.append(val)

    if len(boot_stats) < 10:
        return None, None

    lo = float(np.percentile(boot_stats, alpha / 2 * 100))
    hi = float(np.percentile(boot_stats, (1 - alpha / 2) * 100))
    return lo, hi


def metrics_by_week(preds: list[float], targets: list[float],
                    weeks: list[int],
                    base_preds: Optional[list[float]] = None) -> dict:
    """Métricas agrupadas por semana."""
    by_week = {}
    unique_weeks = sorted(set(weeks))
    for w in unique_weeks:
        idx = [i for i, wk in enumerate(weeks) if wk == w]
        p_w = [preds[i] for i in idx]
        t_w = [targets[i] for i in idx]
        b_w = [base_preds[i] for i in idx] if base_preds else None
        m = core_metrics(p_w, t_w, b_w)
        by_week[str(w)] = {k: v for k, v in m.items() if k != "n"}
    return by_week


def metrics_by_projection_band(preds: list[float], targets: list[float],
                               n_bands: int = 4) -> dict:
    """Métricas agrupadas por faixa de projeção (quartis do valor predito)."""
    if len(preds) < n_bands * 2:
        return {}
    p = np.array(preds, dtype=float)
    t = np.array(targets, dtype=float)
    quantiles = np.quantile(p, np.linspace(0, 1, n_bands + 1))
    bands = []
    for i in range(n_bands):
        lo_q = quantiles[i]
        hi_q = quantiles[i + 1]
        mask = (p >= lo_q) & (p <= hi_q)
        if mask.sum() == 0:
            continue
        p_b = p[mask].tolist()
        t_b = t[mask].tolist()
        m = core_metrics(p_b, t_b)
        bands.append({
            "band": f"{i + 1}",
            "pred_lo": round(float(lo_q), 2),
            "pred_hi": round(float(hi_q), 2),
            "n": int(mask.sum()),
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
        })
    return {"bands": bands}
