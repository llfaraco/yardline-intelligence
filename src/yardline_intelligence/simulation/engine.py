"""Simulação probabilística — v0.3.

Constrói distribuições preditivas a partir de resíduos de previsões
fora da amostra (out-of-sample). Produz média, mediana, P10, P25, P75, P90
e avalia calibração (CRPS, cobertura dos intervalos).

Não apresenta intervalos arbitrários como probabilidades calibradas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ProbabilisticForecast:
    """Distribuição preditiva para um jogador em uma semana."""
    sleeper_id: str
    season: int
    week: int
    position: str
    baseline: float  # previsão do modelo (ponto central)
    residuals: list[float]  # resíduos históricos fora da amostra
    mean: Optional[float] = None
    median: Optional[float] = None
    p10: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "sleeper_id": self.sleeper_id, "season": self.season, "week": self.week,
            "position": self.position, "baseline": self.baseline,
            "n_residuals": len(self.residuals),
            "mean": self.mean, "median": self.median,
            "p10": self.p10, "p25": self.p25, "p75": self.p75, "p90": self.p90,
            "std": self.std,
        }


def build_residual_distribution(residuals: list[float]) -> dict:
    """Constrói distribuição empírica dos resíduos."""
    if not residuals:
        return {}
    r = np.array(residuals, dtype=float)
    return {
        "n": int(len(r)),
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
        "p5": float(np.percentile(r, 5)),
        "p25": float(np.percentile(r, 25)),
        "p50": float(np.percentile(r, 50)),
        "p75": float(np.percentile(r, 75)),
        "p95": float(np.percentile(r, 95)),
        "min": float(np.min(r)),
        "max": float(np.max(r)),
    }


def forecast_from_residuals(sleeper_id: str, season: int, week: int,
                            position: str, baseline: float,
                            residuals: list[float]) -> ProbabilisticForecast:
    """Gera previsão probabilística a partir do baseline + distribuição de resíduos."""
    dist = build_residual_distribution(residuals)
    if not dist:
        return ProbabilisticForecast(
            sleeper_id=sleeper_id, season=season, week=week, position=position,
            baseline=baseline, residuals=[],
        )
    mean = baseline + dist["mean"]
    median = baseline + dist["p50"]
    p10 = baseline + dist["p5"]
    p25 = baseline + dist["p25"]
    p75 = baseline + dist["p75"]
    p90 = baseline + dist["p95"]
    std = dist["std"]
    return ProbabilisticForecast(
        sleeper_id=sleeper_id, season=season, week=week, position=position,
        baseline=baseline, residuals=residuals,
        mean=round(mean, 3), median=round(median, 3),
        p10=round(p10, 3), p25=round(p25, 3),
        p75=round(p75, 3), p90=round(p90, 3),
        std=round(std, 3),
    )


def evaluate_calibration(forecasts: list[ProbabilisticForecast],
                         actuals: list[float]) -> dict:
    """Avalia calibração dos intervalos preditivos."""
    if not forecasts or len(forecasts) != len(actuals):
        return {"n": 0}
    n = len(forecasts)
    covered_p10_p90 = 0
    covered_p25_p75 = 0
    crps_vals = []
    for f, y in zip(forecasts, actuals):
        if f.p10 is None or f.p90 is None:
            continue
        if f.p10 <= y <= f.p90:
            covered_p10_p90 += 1
        if f.p25 is not None and f.p75 is not None and f.p25 <= y <= f.p75:
            covered_p25_p75 += 1
        # CRPS aproximado: largura do intervalo P10-P90 penalizada por descobertura.
        width = f.p90 - f.p10
        # Se o real está dentro, CRPS ~ width/2; se fora, penaliza distância.
        if y < f.p10:
            crps = width + 2 * (f.p10 - y)
        elif y > f.p90:
            crps = width + 2 * (y - f.p90)
        else:
            crps = width / 2.0
        crps_vals.append(crps)
    return {
        "n": n,
        "coverage_p10_p90": round(covered_p10_p90 / n, 4) if n else None,
        "coverage_p25_p75": round(covered_p25_p75 / n, 4) if n else None,
        "mean_crps": round(float(np.mean(crps_vals)), 4) if crps_vals else None,
        "ideal_coverage_p10_p90": 0.8,
        "ideal_coverage_p25_p75": 0.5,
    }
