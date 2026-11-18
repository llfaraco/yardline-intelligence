"""Engenharia de features vetorizada — v0.3.

Calcula features temporais (médias móveis, tendências, eficiência) a partir
das colunas base do dataset parquet, de forma vetorizada com groupby + shift +
rolling. Sem data leakage (apenas semanas anteriores).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features temporais ao dataset base de forma vetorizada.

    Colunas criadas:
    - snap_share_prev, snap_share_mean_3, snap_share_mean_5, snap_share_trend
    - targets_prev, targets_mean_3, targets_mean_5, targets_trend
    - catches_mean_3, catches_mean_5
    - air_yards_mean_3, air_yards_per_target_mean_3
    - epa_per_target_mean_3, epa_sum_mean_3
    - receiving_yards_mean_3, rush_yards_mean_3
    - games_played_prev_n, weeks_active, target_consistency

    Todas usam apenas semanas anteriores (shift >= 1), sem data leakage.
    """
    if df.empty:
        return df

    # Keep each player history inside its own season when datasets are concatenated.
    group_cols = ["sleeper_id", "season"] if "season" in df.columns else ["sleeper_id"]
    sort_cols = ["season", "sleeper_id", "week"] if "season" in df.columns else ["sleeper_id", "week"]
    df = df.sort_values(sort_cols).copy()
    g = df.groupby(group_cols, sort=False)
    group_levels = list(range(len(group_cols)))

    # Snap share.
    if "snap_share" in df.columns:
        df["snap_share"] = pd.to_numeric(df["snap_share"], errors="coerce")
        df["snap_share_prev"] = g["snap_share"].shift(1)
        df["snap_share_mean_3"] = g["snap_share"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df["snap_share_mean_5"] = g["snap_share"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        df["snap_share_trend"] = (
            g["snap_share"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
            - g["snap_share"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean().shift(2))
        )

    # Targets.
    if "targets" in df.columns:
        df["targets"] = pd.to_numeric(df["targets"], errors="coerce")
        df["targets_prev"] = g["targets"].shift(1)
        df["targets_mean_3"] = g["targets"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df["targets_mean_5"] = g["targets"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        df["targets_trend"] = (
            g["targets"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
            - g["targets"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean().shift(2))
        )

    # Catches.
    if "catches" in df.columns:
        df["catches"] = pd.to_numeric(df["catches"], errors="coerce")
        df["catches_mean_3"] = g["catches"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df["catches_mean_5"] = g["catches"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )

    # Air yards.
    if "air_yards" in df.columns:
        df["air_yards"] = pd.to_numeric(df["air_yards"], errors="coerce")
        df["air_yards_mean_3"] = g["air_yards"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        if "targets" in df.columns:
            df["air_yards_per_target_mean_3"] = (
                g.apply(lambda d: (d["air_yards"].shift(1).rolling(3).sum() /
                                    d["targets"].shift(1).rolling(3).sum()),
                        include_groups=False)
                .reset_index(level=group_levels, drop=True)
            )

    # EPA.
    if "epa_sum" in df.columns:
        df["epa_sum"] = pd.to_numeric(df["epa_sum"], errors="coerce")
        df["epa_sum_mean_3"] = g["epa_sum"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        if "targets" in df.columns:
            df["epa_per_target_mean_3"] = (
                g.apply(lambda d: (d["epa_sum"].shift(1).rolling(3).sum() /
                                    d["targets"].shift(1).rolling(3).sum()),
                        include_groups=False)
                .reset_index(level=group_levels, drop=True)
            )

    # Receiving / rush yards.
    if "receiving_yards" in df.columns:
        df["receiving_yards"] = pd.to_numeric(df["receiving_yards"], errors="coerce")
        df["receiving_yards_mean_3"] = g["receiving_yards"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
    if "rush_yards" in df.columns:
        df["rush_yards"] = pd.to_numeric(df["rush_yards"], errors="coerce")
        df["rush_yards_mean_3"] = g["rush_yards"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

    # Jogos / disponibilidade.
    if "games_played" in df.columns:
        df["games_played"] = pd.to_numeric(df["games_played"], errors="coerce")
        df["games_played_prev_n"] = g["games_played"].transform(lambda s: s.shift(1).cumsum())
    df["weeks_active"] = g.cumcount()

    # Consistência de targets (desvio padrão das últimas 5 semanas).
    if "targets" in df.columns:
        df["target_consistency"] = g["targets"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).std()
        )

    return df


VOLUME_FEATURES = [
    "snap_share_prev", "snap_share_mean_3", "snap_share_mean_5",
    "snap_share_trend", "targets_mean_3", "targets_mean_5",
    "targets_prev", "targets_trend", "games_played_prev_n", "weeks_active",
    "target_consistency",
]

AIR_EPA_FEATURES = [
    "air_yards_mean_3", "air_yards_per_target_mean_3",
    "epa_per_target_mean_3", "receiving_yards_mean_3",
    "rush_yards_mean_3", "catches_mean_3",
]

ALL_BASE_FEATURES = VOLUME_FEATURES + AIR_EPA_FEATURES
