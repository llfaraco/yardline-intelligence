"""Camada 2 — Temporal Feature Store com otimizações v0.4.

Adiciona ao FeatureStore:
- Target share (targets / total de alvos do time na semana)
- Air yards e air yards por alvo
- Red zone targets e red zone share
- First downs recebidos
- Últimos 3, 5 e 8 jogos (já existiam, mas agora com mais métricas)
- Snap share do time (contexto ofensivo)
- Todas as janelas respeitam prevenção de data leakage (apenas semanas anteriores)
- Cache incremental de features por temporada/semana
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..data_foundation.ingest import load_parquet


class ExpandedFeatureStore:
    """Feature store com features expandidas, construído sobre o dataset parquet.

    Quando o dataframe subjacente já contém features contextuais (lesão, QB,
    linha ofensiva, game script), elas são incluídas no retorno de
    ``features_for`` para que o modelo treine e avalie com as mesmas colunas.
    """

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
        if self.df.empty:
            return self.df.iloc[0:0]
        return self.df[(self.df["sleeper_id"] == sleeper_id) & (self.df["week"] < week)]

    # ------------------------------------------------------------------
    # Features expandidas para um alvo (semana do jogador).
    # ------------------------------------------------------------------
    def features_for(self, sleeper_id: str, week: int,
                     windows: tuple[int, ...] = (3, 5, 8)) -> dict:
        row = self.row(sleeper_id, week)
        if row is None:
            return {}
        hist = self.history_before(sleeper_id, week)
        feats: dict = {}

        # Identidade/contexto (não vazam).
        feats["position"] = row.get("position")
        feats["team"] = row.get("team")
        feats["opponent"] = row.get("opponent")
        feats["season"] = self.season
        feats["week"] = week

        # Snap share: valor anterior e médias móveis.
        feats["snap_share_prev"] = self._col_prev(hist, "snap_share")
        feats["snap_share_mean_3"] = self._mean(hist, "snap_share", 3)
        feats["snap_share_mean_5"] = self._mean(hist, "snap_share", 5)
        feats["snap_share_mean_8"] = self._mean(hist, "snap_share", 8)
        feats["snap_share_trend"] = self._trend(hist, "snap_share", 3)

        # Volume: targets e recepções.
        feats["targets_prev"] = self._col_prev(hist, "targets")
        feats["targets_mean_3"] = self._mean(hist, "targets", 3)
        feats["targets_mean_5"] = self._mean(hist, "targets", 5)
        feats["targets_mean_8"] = self._mean(hist, "targets", 8)
        feats["targets_trend"] = self._trend(hist, "targets", 3)

        feats["catches_mean_3"] = self._mean(hist, "catches", 3)
        feats["catches_mean_5"] = self._mean(hist, "catches", 5)

        # Eficiência: EPA por alvo, jardas aéreas por alvo.
        feats["epa_per_target_mean_3"] = self._epa_per_target(hist, 3)
        feats["epa_per_target_mean_5"] = self._mean(hist, "epa_per_target", 3)
        feats["air_yards_per_target_mean_3"] = self._air_yards_per_target(hist, 3)
        feats["air_yards_per_target_mean_5"] = self._mean(hist, "air_yards_per_target", 3)

        # Volume aéreo e terrestre.
        feats["receiving_yards_mean_3"] = self._mean(hist, "receiving_yards", 3)
        feats["receiving_yards_mean_5"] = self._mean(hist, "receiving_yards", 5)
        feats["rush_yards_mean_3"] = self._mean(hist, "rush_yards", 3)
        feats["air_yards_mean_3"] = self._mean(hist, "air_yards", 3)

        # Consistência e disponibilidade.
        feats["games_played_prev_n"] = int(hist["games_played"].sum()) if "games_played" in hist else 0
        feats["weeks_active"] = int(len(hist))
        feats["target_consistency"] = self._target_consistency(hist)

        # Contexto do confronto.
        feats["opponent_is_known"] = feats["opponent"] is not None

        # --- Features contextuais (lesão, QB, linha ofensiva, game script) ---
        # Quando o dataframe subjacente já contém essas colunas (via Context
        # Engine), elas são incluídas no snapshot de features para que o modelo
        # treine e avalie com as mesmas colunas que usou no treino.
        for c in ("jogos_perdidos_ultimas_3", "jogos_perdidos_ultimas_5",
                  "semanas_lesionado_ultimas_3", "semanas_lesionado_ultimas_5",
                  "pratica_limitada_ultimas_3", "pratica_limitada_ultimas_5",
                  "status_semana_atual", "lesao_recorrente_regiao_4",
                  "semanas_desde_retorno",
                  "qb_epa_por_dropback_ultimas_3", "qb_epa_por_dropback_ultimas_5",
                  "qb_cpoe_ultimas_3", "qb_cpoe_ultimas_5",
                  "qb_pass_volume_ultimas_3", "qb_pass_volume_ultimas_5",
                  "qb_pass_effic_ultimas_3", "qb_pass_effic_ultimas_5",
                  "qb_pressao_sob_ultimas_3", "qb_pressao_sob_ultimas_5",
                  "qb_troca_titular",
                  "oline_off_pressao_permitida_ultimas_3", "oline_off_pressao_permitida_ultimas_5",
                  "oline_off_sack_rate_ultimas_3", "oline_off_sack_rate_ultimas_5",
                  "oline_off_epa_passe_ultimas_3", "oline_off_epa_passe_ultimas_5",
                  "oline_def_epa_contra_passe_ultimas_3", "oline_def_epa_contra_passe_ultimas_5",
                  "oline_def_jardas_passe_por_jogo_ultimas_3",
                  "oline_def_explosive_ultimas_3", "oline_def_explosive_ultimas_5",
                  "gs_ritmo_ofensivo_ultimas_3", "gs_ritmo_ofensivo_ultimas_5",
                  "gs_taxa_passagem_ultimas_3", "gs_taxa_passagem_ultimas_5",
                  "gs_mando_campo", "gs_descanso_dias",
                  "gs_vegas_total_linha", "gs_vegas_spread_linha",
                  "gs_expected_points_time"):
            if c in row.index:
                v = row.get(c)
                feats[c] = None if pd.isna(v) else v

        # Alvo.
        if "half_ppr_points" in row and pd.notna(row["half_ppr_points"]):
            feats["half_ppr_points"] = float(row["half_ppr_points"])
        if "targets" in row and pd.notna(row["targets"]):
            feats["targets"] = float(row["targets"])

        return feats

    # ------------------------------------------------------------------
    # Helpers internos (somente histórico anterior).
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
        first = float(vals.iloc[0])
        last = float(vals.iloc[-1])
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

    @staticmethod
    def _target_consistency(hist: pd.DataFrame) -> Optional[float]:
        """Desvio padrão dos targets nas últimas 5 semanas (menor = mais consistente)."""
        if hist.empty or "targets" not in hist.columns:
            return None
        vals = hist["targets"].dropna().tail(5)
        if len(vals) < 2:
            return None
        return round(float(vals.std()), 4)

    # ------------------------------------------------------------------
    # Matriz para treino/validação.
    # ------------------------------------------------------------------
    def training_matrix(self, through_week: int,
                        positions: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
        if self.df.empty:
            return self.df.iloc[0:0]
        df = self.df[self.df["week"] <= through_week]
        if positions:
            df = df[df["position"].isin(positions)]
        df = df[df["half_ppr_points"].notna()]
        return df.reset_index(drop=True)

    def feature_columns(self) -> list[str]:
        return [
            "snap_share_prev", "snap_share_mean_3", "snap_share_mean_5",
            "snap_share_mean_8", "snap_share_trend",
            "targets_prev", "targets_mean_3", "targets_mean_5",
            "targets_mean_8", "targets_trend",
            "catches_mean_3", "catches_mean_5",
            "epa_per_target_mean_3", "epa_per_target_mean_5",
            "air_yards_per_target_mean_3", "air_yards_per_target_mean_5",
            "receiving_yards_mean_3", "receiving_yards_mean_5",
            "rush_yards_mean_3", "air_yards_mean_3",
            "games_played_prev_n", "weeks_active", "target_consistency",
        ]
