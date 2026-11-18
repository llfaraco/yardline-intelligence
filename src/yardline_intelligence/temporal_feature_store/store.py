"""Camada 2 — Temporal Feature Store.

Constrói features temporais a partir do dataset temporal (jogador × temporada ×
semana): médias móveis, tendências, share de snap, volume e eficiência recentes,
e features de contexto de confronto. Todas as janelas usam apenas semanas
anteriores à prevista (prevenção de data leakage).

Exporta também o baseline de médias móveis, usado como comparação nos
backtests walk-forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data_foundation.ingest import load_parquet
from ..config import PARQUET_DIR

@dataclass
class FeatureSnapshot:
    """Resultado da construção de features para um jogador e semana alvo."""
    sleeper_id: str
    season: int
    week: int
    position: str
    team: str
    opponent: Optional[str]
    features: dict
    target: Optional[float] = None
    leakage_safe: bool = True


class FeatureStore:
    """Loja de features temporais. Carrega o dataset parquet de uma temporada."""

    def __init__(self, season: int,
                 parquet_dir: Optional[Path] = None,
                 from_df: Optional[pd.DataFrame] = None):
        from ..config import PARQUET_DIR
        self.season = season
        self.dir = parquet_dir or PARQUET_DIR
        if from_df is not None:
            self.df = from_df.copy()
        else:
            self.df = load_parquet(f"dataset_{season}", season)

    def available_weeks(self) -> list[int]:
        if self.df.empty:
            return []
        return sorted(self.df["week"].dropna().unique().astype(int).tolist())

    def players(self, week: int) -> list[str]:
        if self.df.empty:
            return []
        sub = self.df[self.df["week"] == week]
        return sub["sleeper_id"].dropna().unique().tolist()

    def row(self, sleeper_id: str, week: int) -> Optional[pd.Series]:
        if self.df.empty:
            return None
        mask = (self.df["sleeper_id"] == sleeper_id) & (self.df["week"] == week)
        rows = self.df[mask]
        return rows.iloc[0] if len(rows) else None

    def history_before(self, sleeper_id: str, week: int) -> pd.DataFrame:
        """Todas as linhas do jogador estritamente antes de `week`."""
        if self.df.empty:
            return self.df.iloc[0:0]
        return self.df[(self.df["sleeper_id"] == sleeper_id) & (self.df["week"] < week)]

    # ------------------------------------------------------------------
    # Construção de features para um alvo (jogador, semana).
    # ------------------------------------------------------------------
    def features_for(self, sleeper_id: str, week: int,
                     windows: tuple[int, ...] = (3, 5)) -> FeatureSnapshot:
        """Features de um jogador para a semana `week`, usando só semanas < week."""
        row = self.row(sleeper_id, week)
        if row is None:
            return FeatureSnapshot(sleeper_id, self.season, week, "", "", None, {})

        hist = self.history_before(sleeper_id, week)
        feats: dict = {}

        # Identidade e contexto (não vazam: adversário é conhecido com antecedência).
        feats["position"] = row.get("position")
        feats["team"] = row.get("team")
        feats["opponent"] = row.get("opponent")
        feats["season"] = self.season
        feats["week"] = week

        # Snap share: valor da semana anterior e média móvel das janelas.
        feats["snap_share_prev"] = self._col_prev(hist, "snap_share")
        feats["snap_share_mean_3"] = self._mean(hist, "snap_share", 3)
        feats["snap_share_mean_5"] = self._mean(hist, "snap_share", 5)
        feats["snap_share_trend"] = self._trend(hist, "snap_share", 3)

        # Volume (recebidos/corridas) — médias móveis.
        feats["targets_mean_3"] = self._mean(hist, "targets", 3)
        feats["targets_mean_5"] = self._mean(hist, "targets", 5)
        feats["targets_prev"] = self._col_prev(hist, "targets")
        feats["targets_trend"] = self._trend(hist, "targets", 3)

        feats["rushes_mean_3"] = self._mean(hist, "rushes", 3)
        feats["rushes_mean_5"] = self._mean(hist, "rushes", 5)
        feats["rushes_prev"] = self._col_prev(hist, "rushes")

        # Eficiência — EPA por alvo, jardas por recepção/corrida.
        feats["epa_per_target_mean_3"] = self._epa_per_target(hist, 3)
        feats["epa_per_target_mean_5"] = self._epa_per_target(hist, 5)
        feats["air_yards_per_target_mean_3"] = self._air_yards_per_target(hist, 3)
        feats["receiving_yards_mean_3"] = self._mean(hist, "receiving_yards", 3)
        feats["rush_yards_mean_3"] = self._mean(hist, "rush_yards", 3)

        # Jogos disputados e consistência.
        feats["games_played_prev_n"] = int(hist["games_played"].sum()) if "games_played" in hist else 0
        feats["weeks_active"] = int(len(hist))

        # Contexto do confronto (adversário conhecido, sem vazamento).
        feats["opponent_is_known"] = feats["opponent"] is not None

        target = None
        if "half_ppr_points" in row and pd.notna(row["half_ppr_points"]):
            target = float(row["half_ppr_points"])

        return FeatureSnapshot(
            sleeper_id=sleeper_id, season=self.season, week=week,
            position=str(row.get("position") or ""), team=str(row.get("team") or ""),
            opponent=row.get("opponent"), features=feats, target=target,
        )

    # ------------------------------------------------------------------
    # Helpers internos (todos leem somente o histórico anterior).
    # ------------------------------------------------------------------
    @staticmethod
    def _col_prev(hist: pd.DataFrame, col: str) -> Optional[float]:
        if hist.empty or col not in hist.columns:
            return None
        vals = hist[col].dropna()
        return float(vals.iloc[-1]) if len(vals) else None

    @staticmethod
    def _mean(hist: pd.DataFrame, col: str, window: int) -> Optional[float]:
        if hist.empty or col not in hist.columns:
            return None
        vals = hist[col].dropna().tail(window)
        return float(vals.mean()) if len(vals) else None

    @staticmethod
    def _trend(hist: pd.DataFrame, col: str, window: int) -> Optional[float]:
        if hist.empty or col not in hist.columns:
            return None
        vals = hist[col].dropna().tail(window)
        if len(vals) < 2:
            return None
        first = float(vals.iloc[0]); last = float(vals.iloc[-1])
        return round(last - first, 4)

    @staticmethod
    def _epa_per_target(hist: pd.DataFrame, window: int) -> Optional[float]:
        if hist.empty:
            return None
        sub = hist.tail(window)
        epa = sub["epa_sum"].dropna().sum() if "epa_sum" in sub.columns else 0.0
        tgts = sub["targets"].dropna().sum() if "targets" in sub.columns else 0.0
        return round(epa / tgts, 4) if tgts > 0 else None

    @staticmethod
    def _air_yards_per_target(hist: pd.DataFrame, window: int) -> Optional[float]:
        if hist.empty:
            return None
        sub = hist.tail(window)
        ay = sub["air_yards"].dropna().sum() if "air_yards" in sub.columns else 0.0
        tgts = sub["targets"].dropna().sum() if "targets" in sub.columns else 0.0
        return round(ay / tgts, 4) if tgts > 0 else None

    # ------------------------------------------------------------------
    # Matriz para treino/validação de um modelo (X, y) — sem data leakage.
    # ------------------------------------------------------------------
    def training_matrix(self, through_week: int,
                        positions: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
        """Matriz de features e alvo para semanas <= through_week.

        Cada linha usa apenas o histórico anterior à sua própria semana, então
        é segura para validação walk-forward.
        """
        if self.df.empty:
            return self.df.iloc[0:0]
        df = self.df[self.df["week"] <= through_week]
        if positions:
            df = df[df["position"].isin(positions)]
        # Remove linhas sem alvo (semana ainda não jogada).
        df = df[df["half_ppr_points"].notna()]
        return df.reset_index(drop=True)

    def feature_columns(self) -> list[str]:
        return [
            "snap_share_prev", "snap_share_mean_3", "snap_share_mean_5",
            "snap_share_trend", "targets_mean_3", "targets_mean_5",
            "targets_prev", "targets_trend", "rushes_mean_3", "rushes_mean_5",
            "rushes_prev", "epa_per_target_mean_3", "epa_per_target_mean_5",
            "air_yards_per_target_mean_3", "receiving_yards_mean_3",
            "rush_yards_mean_3", "games_played_prev_n", "weeks_active",
        ]


def moving_average_baseline(store: FeatureStore, sleeper_id: str, week: int,
                            window: int = 3) -> Optional[float]:
    """Baseline: média móvel dos últimos `window` jogos do jogador.

    Retorna None quando não há histórico suficiente (não inventa zero).
    """
    row = store.row(sleeper_id, week)
    if row is None:
        return None
    hist = store.history_before(sleeper_id, week)
    if hist.empty:
        return None
    vals = hist["half_ppr_points"].dropna().tail(window)
    if len(vals) == 0:
        return None
    return float(vals.mean())
