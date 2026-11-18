"""Contexto do quarterback — Camada do Context Engine.

Constrói, para cada time e semana, features derivadas do play-by-play que
descrevem o desempenho do QB titular do time, usando APENAS semanas anteriores
à previsão (sem data leakage).

Essas features são atribuídas a todos os WRs/TEs/RBs do time, pois o contexto
do QB afeta o volume e eficiência de todos os receptores/corredores.

Features:
- qb_epa_por_dropback_ultimas_N: média de EPA por dropback do QB
- qb_cpoe_ultimas_N: CPOE (Completed Probability Over Expectation) médio
- qb_pass_volume_ultimas_N: número de passes tentados
- qb_pass_effic_ultimas_N: pontos Half-PPR de passe por dropback
- qb_pressao_sob_ultimas_N: taxa de pressão sobre o QB (sack + qb_hit por dropback)
- qb_troca_titular: se o QB mudou de titular na semana (comparado com semana anterior)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..data_foundation.ingest import read_csv_cached, CACHE_DIR
from ..config import CURRENT_SEASON

PBP_QB_COLUMNS = [
    "season", "week", "game_type", "game_id", "posteam", "qb_dropback", "pass_attempt",
    "complete_pass", "epa", "cpoe", "passer_player_id", "passer_player_name",
    "qb_hit", "sack", "interception", "pass_td", "passing_yards",
]


def load_pbp_qb(season: int,
                cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Carrega colunas do pbp relevantes ao QB de uma temporada."""
    path = cache_dir / f"pbp_{season}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    return read_csv_cached(__import__("duckdb").connect(),
                           path, PBP_QB_COLUMNS)


def _is_qb_dropback(row) -> bool:
    return str(row.get("qb_dropback", "0")) in ("1", "1.0")


def build_qb_context(pbp: pd.DataFrame,
                     team_to_weeks_qb: dict,
                     season: int,
                     max_week: int) -> pd.DataFrame:
    """Constrói features de contexto do QB por (time, week).

    team_to_weeks_qb: mapeia time -> {week: passer_id} (QB titular conhecido
    antecipadamente, vindo do calendário/elenco). Quando não disponível, usa o
    QB com mais dropbacks na semana anterior.

    Retorna DataFrame com colunas: team, season, week, e as features de QB.
    """
    if pbp.empty:
        return pd.DataFrame(columns=[
            "team", "season", "week",
            "qb_epa_por_dropback_ultimas_3", "qb_epa_por_dropback_ultimas_5",
            "qb_cpoe_ultimas_3", "qb_cpoe_ultimas_5",
            "qb_pass_volume_ultimas_3", "qb_pass_volume_ultimas_5",
            "qb_pass_effic_ultimas_3", "qb_pass_effic_ultimas_5",
            "qb_pressao_sob_ultimas_3", "qb_pressao_sob_ultimas_5",
            "qb_troca_titular",
        ])

    p = pbp.copy()
    p["week"] = pd.to_numeric(p["week"], errors="coerce")
    p["season"] = pd.to_numeric(p["season"], errors="coerce")
    if "game_type" in p.columns:
        p = p[(p["season"] == season) & (p["game_type"] == "REG")]
    elif "season_type" in p.columns:
        p = p[(p["season"] == season) & (p["season_type"] == "REG")]
    else:
        p = p[p["season"] == season]
    p = p.dropna(subset=["week", "posteam"])
    p["week"] = p["week"].astype(int)

    # Normaliza flags numéricas.
    for col in ("qb_dropback", "pass_attempt", "complete_pass", "qb_hit", "sack",
                "interception", "pass_td"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0).astype(int)
    for col in ("epa", "cpoe", "passing_yards"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    p["is_dropback"] = p.apply(_is_qb_dropback, axis=1)

    # Determina QB titular por time/semana (mais dropbacks na semana).
    qb_by_team_week = {}
    for (team, week), sub in p.groupby(["posteam", "week"]):
        dropbacks = sub[sub["is_dropback"] == 1]
        if len(dropbacks):
            qb_id = dropbacks["passer_player_id"].mode()
            if len(qb_id):
                qb_by_team_week[(team, week)] = str(qb_id.iloc[0])

    rows = []
    teams = sorted(p["posteam"].unique())
    for team in teams:
        for week in range(1, max_week + 1):
            prior = p[(p["posteam"] == team) & (p["week"] < week)]
            if prior.empty:
                rows.append({
                    "team": team, "season": season, "week": week,
                    "qb_epa_por_dropback_ultimas_3": None,
                    "qb_epa_por_dropback_ultimas_5": None,
                    "qb_cpoe_ultimas_3": None,
                    "qb_cpoe_ultimas_5": None,
                    "qb_pass_volume_ultimas_3": None,
                    "qb_pass_volume_ultimas_5": None,
                    "qb_pass_effic_ultimas_3": None,
                    "qb_pass_effic_ultimas_5": None,
                    "qb_pressao_sob_ultimas_3": None,
                    "qb_pressao_sob_ultimas_5": None,
                    "qb_troca_titular": None,
                })
                continue

            feats = {"team": team, "season": season, "week": week}

            for wn, epa_col in [(3, "qb_epa_por_dropback_ultimas_3"),
                                (5, "qb_epa_por_dropback_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db) and "epa" in db.columns and db["epa"].notna().any():
                    feats[epa_col] = round(float(db["epa"].mean()), 4)
                else:
                    feats[epa_col] = None

            for wn, cpoe_col in [(3, "qb_cpoe_ultimas_3"),
                                 (5, "qb_cpoe_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db) and "cpoe" in db.columns and db["cpoe"].notna().any():
                    feats[cpoe_col] = round(float(db["cpoe"].mean()), 4)
                else:
                    feats[cpoe_col] = None

            for wn, vol_col in [(3, "qb_pass_volume_ultimas_3"),
                                (5, "qb_pass_volume_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                feats[vol_col] = int((recent["pass_attempt"] == 1).sum()) \
                    if len(recent) > 0 else None

            for wn, eff_col in [(3, "qb_pass_effic_ultimas_3"),
                                (5, "qb_pass_effic_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db):
                    # Half-PPR de passe: 0.04 por jarda + 4 por TD - 1 por INT.
                    yards = db["passing_yards"].sum() if "passing_yards" in db.columns else 0.0
                    tds = int((db["pass_td"] == 1).sum()) if "pass_td" in db.columns else 0
                    ints = int((db["interception"] == 1).sum()) if "interception" in db.columns else 0
                    pts = 0.04 * float(yards) + 4.0 * tds - 1.0 * ints
                    feats[eff_col] = round(pts / max(len(db), 1), 4)
                else:
                    feats[eff_col] = None

            for wn, press_col in [(3, "qb_pressao_sob_ultimas_3"),
                                  (5, "qb_pressao_sob_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db):
                    press = int(((db["qb_hit"] == 1) | (db["sack"] == 1)).sum())
                    feats[press_col] = round(press / len(db), 4)
                else:
                    feats[press_col] = None

            # Troca de titular: QB da semana anterior diferente do QB atual.
            atual_qb = qb_by_team_week.get((team, week))
            ant_semana = week - 1
            ant_qb = qb_by_team_week.get((team, ant_semana))
            if atual_qb is None or ant_qb is None:
                feats["qb_troca_titular"] = None
            else:
                feats["qb_troca_titular"] = int(atual_qb != ant_qb)

            rows.append(feats)

    return pd.DataFrame(rows)
