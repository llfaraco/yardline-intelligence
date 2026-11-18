"""Context Engine — orquestrador das features contextuais.

Integra as quatro famílias de variáveis ao dataset temporal de jogadores:
- Disponibilidade e lesões (injury_context)
- Quarterback (qb_context)
- Linha ofensiva / pass rush / secundária (oline_context)
- Game script (gamescript_context)

Todas as features usam apenas semanas anteriores à previsão (sem data leakage)
e são atribuídas ao jogador pelo seu time e pela semana alvo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import CACHE_DIR, CURRENT_SEASON
from ..data_foundation.dataset import build_player_dataset, _to_num
from .injury_context import load_injuries, build_injury_features
from .qb_context import load_pbp_qb, build_qb_context
from .oline_context import load_pbp_oline, build_oline_context
from .gamescript_context import load_pbp_gamescript, build_gamescript_context


class ContextEngine:
    """Constrói e mescla features contextuais ao dataset temporal de jogadores."""

    def __init__(self, season: int,
                 cache_dir: Path = CACHE_DIR,
                 parquet_dir: Optional[Path] = None,
                 from_df: Optional[pd.DataFrame] = None):
        self.season = season
        self.cache_dir = cache_dir
        self.parquet_dir = parquet_dir
        self.from_df = from_df

    def build_full_dataset(self,
                           max_week: Optional[int] = None) -> pd.DataFrame:
        """Constrói o dataset temporal com todas as features contextuais.

        Retorna DataFrame com colunas do dataset base + features contextuais.
        """
        # 1. Dataset base (jogador × semana, com targets e features de volume).
        if self.from_df is not None:
            df_base = self.from_df.copy()
        else:
            df_base = build_player_dataset(self.season,
                                           cache_dir=self.cache_dir,
                                           parquet_dir=self.parquet_dir)
        if df_base.empty:
            return df_base

        if max_week is None:
            max_week = int(df_base["week"].max())

        # 2. Mapeamentos GSIS <-> Sleeper (precisamos reconstruir para features de lesão/QB).
        gsis_to_sleeper, sleeper_to_gsis, team_to_players = self._build_id_maps(df_base)

        # 3. Disponibilidade e lesões.
        inj_df = load_injuries(self.season, cache_dir=self.cache_dir)
        inj_features = build_injury_features(inj_df, gsis_to_sleeper, sleeper_to_gsis,
                                             self.season, max_week)

        # 4. Contexto do QB por time/semana.
        pbp_qb = load_pbp_qb(self.season, cache_dir=self.cache_dir)
        qb_context = build_qb_context(pbp_qb, {}, self.season, max_week)

        # 5. Linha ofensiva / pass rush / secundária.
        pbp_oline = load_pbp_oline(self.season, cache_dir=self.cache_dir)
        oline_context = build_oline_context(pbp_oline, self.season, max_week)

        # 6. Game script.
        pbp_gs = load_pbp_gamescript(self.season, cache_dir=self.cache_dir)
        gs_context = build_gamescript_context(pbp_gs, self.season, max_week)

        # 7. Mescla tudo no dataset base (por sleeper_id + week).
        df = df_base.copy()

        # Lesão: mesclar por sleeper_id + week.
        if not inj_features.empty:
            inj_features["season"] = self.season
            df = df.merge(inj_features,
                          on=["sleeper_id", "season", "week"],
                          how="left")

        # Para features por time, precisamos mapear jogador -> time da semana.
        # oline_context tem ofensa e defesa; usamos ofensa para o time do jogador
        # e defesa para o adversário.
        oline_off = oline_context[oline_context["lado"] == "ofensa"].copy()
        oline_def = oline_context[oline_context["lado"] == "defesa"].copy()

        # Renomeia TODAS as colunas com prefixo para evitar colisão no merge.
        oline_off = oline_off.rename(columns={c: f"oline_off_{c}" for c in oline_off.columns})
        oline_def = oline_def.rename(columns={c: f"oline_def_{c}" for c in oline_def.columns})

        df = df.merge(oline_off,
                      left_on=["team", "season", "week"],
                      right_on=["oline_off_team", "oline_off_season", "oline_off_week"],
                      how="left")
        df = df.merge(oline_def,
                      left_on=["opponent", "season", "week"],
                      right_on=["oline_def_team", "oline_def_season", "oline_def_week"],
                      how="left")

        # QB context: mesclar pelo time do jogador (ofensa).
        # qb_context já tem colunas sem prefixo (team, season, week, qb_epa_...).
        # Renomeia só team/season/week para evitar colisão, mantém qb_* intacto.
        qb_renamed = {"team": "qb_team", "season": "qb_season", "week": "qb_week"}
        qb_context = qb_context.rename(columns=qb_renamed)
        df = df.merge(qb_context,
                      left_on=["team", "season", "week"],
                      right_on=["qb_team", "qb_season", "qb_week"],
                      how="left")

        # Game script: mesclar pelo time do jogador.
        gs_renamed = {"team": "gs_team", "season": "gs_season", "week": "gs_week"}
        gs_context = gs_context.rename(columns=gs_renamed)
        df = df.merge(gs_context,
                      left_on=["team", "season", "week"],
                      right_on=["gs_team", "gs_season", "gs_week"],
                      how="left")

        # Limpa colunas duplicadas/auxiliares de merge.
        drop_cols = [c for c in df.columns if c.endswith("_team") or c.endswith("_season")
                     or c.endswith("_week") or c.endswith("_lado")]
        # Preserva as colunas originais team/season/week.
        for c in list(drop_cols):
            if c in ("team", "season", "week"):
                drop_cols.remove(c)
        df = df.drop(columns=drop_cols, errors="ignore")

        return df

    def _build_id_maps(self, df_base: pd.DataFrame):
        """Reconstrói mapas GSIS<->Sleeper a partir do dataset base."""
        gsis_to_sleeper = {}
        sleeper_to_gsis = {}
        team_to_players = {}
        if "gsis_id" in df_base.columns and "sleeper_id" in df_base.columns:
            for _, r in df_base.dropna(subset=["gsis_id", "sleeper_id"]).iterrows():
                gsis_to_sleeper[str(r["gsis_id"])] = str(r["sleeper_id"])
                sleeper_to_gsis[str(r["sleeper_id"])] = str(r["gsis_id"])
        if "team" in df_base.columns:
            team_to_players = df_base.groupby("team")["sleeper_id"].apply(set).to_dict()
        return gsis_to_sleeper, sleeper_to_gsis, team_to_players
