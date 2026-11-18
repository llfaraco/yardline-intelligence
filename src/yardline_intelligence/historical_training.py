"""Yardline Intelligence v0.3 — Treinamento Histórico e Avaliação Temporal.

Orquestra o treinamento usando temporadas históricas (2018-2023), seleção
de modelo em 2024 e teste retrospectivo intocado em 2025, com diagnóstico
adicional em 2026. Sem divisão aleatória: avaliação estritamente temporal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from yardline_intelligence.config import (  # type: ignore[import-not-found]
    CACHE_DIR, CURRENT_SEASON, EXPERIMENTS_DIR, PARQUET_DIR,
)
from yardline_intelligence.context_engine.context_engine_builder import ContextEngine
from yardline_intelligence.data_foundation.dataset import build_player_dataset
from yardline_intelligence.data_foundation.ingest import load_parquet


def load_season_dataset(season: int,
                        max_week: Optional[int] = None) -> pd.DataFrame:
    """Carrega o dataset base de uma temporada (sem features contextuais)."""
    path = PARQUET_DIR / str(season) / f"dataset_{season}.parquet"
    if path.exists():
        df = load_parquet(f"dataset_{season}", season)
    else:
        df = build_player_dataset(season, cache_dir=CACHE_DIR, parquet_dir=PARQUET_DIR)
    if max_week is not None and not df.empty:
        df = df[df["week"] <= max_week].copy()
    return df


def build_season_with_context(season: int,
                              max_week: Optional[int] = None) -> pd.DataFrame:
    """Constrói dataset completo (base + contexto) de uma temporada."""
    ce = ContextEngine(season=season, cache_dir=CACHE_DIR, parquet_dir=PARQUET_DIR)
    return ce.build_full_dataset(max_week=max_week)


def temporal_train_validate_test(seasons_train: list[int],
                                 season_select: int,
                                 season_test: int,
                                 max_week_test: Optional[int] = None) -> dict:
    """Executa treinamento/validação/teste estritamente temporal.

    - Treino: temporadas `seasons_train` (todas as semanas).
    - Seleção: `season_select` (ajusta hiperparâmetros).
    - Teste: `season_test` (intocado, usado apenas para avaliação final).

    Retorna dicionário com os datasets e metadados de cada etapa.
    """
    result = {
        "seasons_train": seasons_train,
        "season_select": season_select,
        "season_test": season_test,
        "train": {},
        "select": {},
        "test": {},
        "notes": [],
    }

    # ------------------------------------------------------------------
    # Treino: concatena temporadas históricas.
    # ------------------------------------------------------------------
    train_frames = []
    for s in seasons_train:
        df = load_season_dataset(s)
        if not df.empty:
            train_frames.append(df)
            result["train"][str(s)] = {
                "rows": int(len(df)),
                "weeks": sorted(df["week"].dropna().unique().astype(int).tolist()),
                "players": int(df["sleeper_id"].nunique()),
            }
    if train_frames:
        df_train = pd.concat(train_frames, ignore_index=True)
        result["train"]["combined"] = {
            "rows": int(len(df_train)),
            "players": int(df_train["sleeper_id"].nunique()),
            "seasons": seasons_train,
        }
        result["train"]["df"] = df_train
    else:
        result["train"]["df"] = pd.DataFrame()
        result["notes"].append("Nenhum dado de treino disponível")

    # ------------------------------------------------------------------
    # Seleção: temporada usada para ajustar hiperparâmetros.
    # ------------------------------------------------------------------
    df_select = load_season_dataset(season_select)
    if not df_select.empty:
        result["select"] = {
            "rows": int(len(df_select)),
            "weeks": sorted(df_select["week"].dropna().unique().astype(int).tolist()),
            "players": int(df_select["sleeper_id"].nunique()),
        }
        result["select"]["df"] = df_select
    else:
        result["select"]["df"] = pd.DataFrame()
        result["notes"].append(f"Temporada de seleção {season_select} não disponível")

    # ------------------------------------------------------------------
    # Teste: temporada intocado (apenas avaliação final).
    # ------------------------------------------------------------------
    df_test = load_season_dataset(season_test, max_week=max_week_test)
    if not df_test.empty:
        result["test"] = {
            "rows": int(len(df_test)),
            "weeks": sorted(df_test["week"].dropna().unique().astype(int).tolist()),
            "players": int(df_test["sleeper_id"].nunique()),
            "max_week": max_week_test,
        }
        result["test"]["df"] = df_test
    else:
        result["test"]["df"] = pd.DataFrame()
        result["notes"].append(f"Temporada de teste {season_test} não disponível")

    return result


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    res = temporal_train_validate_test(
        seasons_train=[2018, 2019, 2020, 2021, 2022, 2023],
        season_select=2024,
        season_test=2025,
    )
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("df",)}, indent=2, ensure_ascii=False))
    print(f"\nTreino combinado: {len(res['train']['df'])} linhas")
    print(f"Seleção: {len(res['select']['df'])} linhas")
    print(f"Teste: {len(res['test']['df'])} linhas")
