"""Contexto de disponibilidade e lesões — Camada do Context Engine.

Constrói, para cada jogador e semana, features derivadas do histórico de
lesões (injuries_<season>.csv do nflverse), usando APENAS semanas anteriores
à previsão (sem data leakage).

Features:
- jogos_perdidos_ultimas_N: semanas em que o atleta ficou fora (report_status=Out/IR/PUP)
- semanas_lesionado_ultimas_N: semanas com qualquer lesão relatada
- pratica_limitada_ultimas_N: semanas com treino limitado (practice_status != full)
- status_semana_atual: status reportado para a semana alvo (conhecido antecipadamente)
- lesao_recorrente_regiao: se teve lesão na mesma região corporal nas últimas 4 semanas
- semanas_desde_retorno: semanas desde o último retorno de lesão grave
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..data_foundation.ingest import read_csv_cached, CACHE_DIR
from ..config import CURRENT_SEASON

# Status que indicam ausência completa do jogo.
OUT_STATUSES = {"out", "ir", "pup", "sus", "suspended", "injured reserve",
                "physically unable", "na", "dnr", "did not report"}
# Status de lesão (qualquer relato, incluindo questionable/doubtful).
INJURY_STATUSES = OUT_STATUSES | {"doubtful", "questionable"}
# Status de treino que indica limitação.
LIMITED_PRACTICE = {"limited", "did not participate", "dnp", "out"}


def _status_value(v: Optional[str]) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    return s if s else None


def load_injuries(season: int,
                  cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Carrega o histórico de lesões de uma temporada."""
    path = cache_dir / f"injuries_{season}.csv"
    if not path.exists():
        return pd.DataFrame()
    return read_csv_cached(__import__("duckdb").connect(),
                           path,
                           ["season", "season_type", "game_type", "team", "week",
                            "gsis_id", "position", "full_name",
                            "report_primary_injury", "report_secondary_injury",
                            "report_status",
                            "practice_primary_injury", "practice_secondary_injury",
                            "practice_status"])


def build_injury_features(injuries: pd.DataFrame,
                          gsis_to_sleeper: dict,
                          sleeper_to_gsis: dict,
                          season: int,
                          max_week: int) -> pd.DataFrame:
    """Constrói features de lesão por (sleeper_id, week).

    Retorna DataFrame com colunas: sleeper_id, season, week, e as features
    de disponibilidade. Usa apenas semanas < week (sem data leakage).
    """
    if injuries.empty:
        return pd.DataFrame(columns=[
            "sleeper_id", "season", "week",
            "jogos_perdidos_ultimas_3", "jogos_perdidos_ultimas_5",
            "semanas_lesionado_ultimas_3", "semanas_lesionado_ultimas_5",
            "pratica_limitada_ultimas_3", "pratica_limitada_ultimas_5",
            "status_semana_atual", "lesao_recorrente_regiao_4",
            "semanas_desde_retorno",
        ])

    inj = injuries.copy()
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj = inj[inj["season"] == season]
    if "game_type" in inj.columns:
        inj = inj[inj["game_type"] == "REG"]
    elif "season_type" in inj.columns:
        inj = inj[inj["season_type"] == "REG"]
    inj = inj.dropna(subset=["week"])
    inj["week"] = inj["week"].astype(int)

    # Traduz gsis_id para sleeper_id.
    inj["sleeper_id"] = inj["gsis_id"].astype(str).map(gsis_to_sleeper)
    inj = inj.dropna(subset=["sleeper_id"])

    # Mapeia status.
    inj["report_status_norm"] = inj["report_status"].map(_status_value)
    inj["practice_status_norm"] = inj["practice_status"].map(_status_value)

    # Semana do lesão (região corporal) para recorrência.
    inj["injury_region"] = inj["report_primary_injury"].map(_status_value)

    rows = []
    sleeper_ids = sorted(inj["sleeper_id"].unique())
    for sid in sleeper_ids:
        sub = inj[inj["sleeper_id"] == sid].sort_values("week")
        for week in range(1, max_week + 1):
            prior = sub[sub["week"] < week]
            if prior.empty:
                # Sem histórico de lesão conhecido: ausência, não zero.
                rows.append({
                    "sleeper_id": sid, "season": season, "week": week,
                    "jogos_perdidos_ultimas_3": None,
                    "jogos_perdidos_ultimas_5": None,
                    "semanas_lesionado_ultimas_3": None,
                    "semanas_lesionado_ultimas_5": None,
                    "pratica_limitada_ultimas_3": None,
                    "pratica_limitada_ultimas_5": None,
                    "status_semana_atual": None,
                    "lesao_recorrente_regiao_4": None,
                    "semanas_desde_retorno": None,
                })
                continue

            # Jogos perdidos nas últimas 3/5 semanas (semana atual excluída).
            for wn, col in [(3, "jogos_perdidos_ultimas_3"),
                            (5, "jogos_perdidos_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                perdidos = int((recent["report_status_norm"].isin(OUT_STATUSES)).sum())
                # Se não há linhas na semana, considerar como não jogou (bye ou sem dados).
                # Usar None quando não há dados suficientes.
                rows_tmp = {}

            # Recalcula limpo.
            feats = {"sleeper_id": sid, "season": season, "week": week}
            for wn, col in [(3, "jogos_perdidos_ultimas_3"),
                            (5, "jogos_perdidos_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                feats[col] = int((recent["report_status_norm"].isin(OUT_STATUSES)).sum()) \
                    if len(recent) > 0 else None

            for wn, col in [(3, "semanas_lesionado_ultimas_3"),
                            (5, "semanas_lesionado_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                feats[col] = int((recent["report_status_norm"].isin(INJURY_STATUSES)).sum()) \
                    if len(recent) > 0 else None

            for wn, col in [(3, "pratica_limitada_ultimas_3"),
                            (5, "pratica_limitada_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                feats[col] = int((recent["practice_status_norm"].isin(LIMITED_PRACTICE)).sum()) \
                    if len(recent) > 0 else None

            # Status da semana atual (conhecido antecipadamente).
            atual = sub[sub["week"] == week]
            if len(atual):
                st = _status_value(atual.iloc[0]["report_status_norm"])
                feats["status_semana_atual"] = st
            else:
                feats["status_semana_atual"] = None

            # Lesão recorrente na mesma região nas últimas 4 semanas.
            reg4 = prior[prior["week"] >= week - 4]
            reg_regions = reg4["injury_region"].dropna().unique()
            feats["lesao_recorrente_regiao_4"] = int(len(reg_regions) > 1)

            # Semanas desde o último retorno (semana sem lesão após lesão).
            # Última semana com lesão antes de `week`.
            lesadas = prior[prior["report_status_norm"].isin(INJURY_STATUSES)]
            if len(lesadas):
                ultima_lesao = int(lesadas["week"].max())
                # Primeira semana sem lesão após a última lesão.
                apos = prior[(prior["week"] > ultima_lesao) &
                             (~prior["report_status_norm"].isin(INJURY_STATUSES))]
                if len(apos):
                    feats["semanas_desde_retorno"] = week - int(apos["week"].min())
                else:
                    feats["semanas_desde_retorno"] = None
            else:
                feats["semanas_desde_retorno"] = None

            rows.append(feats)

    return pd.DataFrame(rows)
