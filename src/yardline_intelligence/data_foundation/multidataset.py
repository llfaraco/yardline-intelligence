"""Constrói dataset temporal unificado de múltiplas temporadas.

Agrega os parquets gerados por build_player_dataset de cada temporada em um
único DataFrame, preservando a coluna season para treino walk-forward temporal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import PARQUET_DIR


def build_multi_season_dataset(seasons: list[int],
                               parquet_dir: Optional[Path] = None) -> pd.DataFrame:
    """Carrega e concatena datasets de várias temporadas."""
    parquet_dir = parquet_dir or PARQUET_DIR
    frames = []
    for season in seasons:
        path = parquet_dir / str(season) / f"dataset_{season}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["season", "sleeper_id", "week"]).reset_index(drop=True)
