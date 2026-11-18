"""Contexto de linha ofensiva, pass rush e secundária — Camada do Context Engine.

Constrói, para cada time e semana, features derivadas do play-by-play que
descrevem a qualidade da proteção de passe, a pressão produzida e a defesa
contra o passe, usando APENAS semanas anteriores à previsão (sem data leakage).

Essas features são atribuídas a todos os WRs/TEs de um time (linha ofensiva e
secundária do adversário afetam o jogo do passe).

Features (ofensa do time do jogador):
- oline_pressao_permitida_ultimas_N: taxa de pressão permitida sobre o QB do time
- oline_sack_rate_ultimas_N: sacks permitidos por dropback
- oline_epa_protecao_ultimas_N: EPA de passe por dropback do time

Features (defesa do adversário):
- def_epa_contra_passe_ultimas_N: EPA permitido por jogada de passe pela defesa adversária
- def_jardas_passe_por_jogo_ultimas_N: jardas aéreas permitidas por jogo
- def_explosive_ultimas_N: explosive plays (>=20 jardas de passe) permitidas
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..data_foundation.ingest import read_csv_cached, CACHE_DIR
from ..config import CURRENT_SEASON

PBP_OLINE_COLUMNS = [
    "season", "week", "game_type", "game_id", "posteam", "defteam", "qb_dropback",
    "qb_hit", "sack", "epa", "passer_player_id",
    "interception", "pass_td", "air_yards", "receiving_yards",
    "play_type", "yards_gained",
]


def load_pbp_oline(season: int,
                   cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Carrega colunas do pbp relevantes à linha ofensiva/defesa de passe."""
    path = cache_dir / f"pbp_{season}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    return read_csv_cached(__import__("duckdb").connect(),
                           path, PBP_OLINE_COLUMNS)


def build_oline_context(pbp: pd.DataFrame,
                        season: int,
                        max_week: int) -> pd.DataFrame:
    """Constrói features de linha ofensiva/defesa por (time, week).

    Retorna DataFrame com colunas: team, season, week, lado ('ofensa' ou 'defesa'),
    e as features de proteção/pressão/secundária.
    """
    if pbp.empty:
        return pd.DataFrame(columns=[
            "team", "season", "week", "lado",
            "pressao_permitida_ultimas_3", "pressao_permitida_ultimas_5",
            "sack_rate_ultimas_3", "sack_rate_ultimas_5",
            "epa_passe_ultimas_3", "epa_passe_ultimas_5",
            "epa_contra_passe_ultimas_3", "epa_contra_passe_ultimas_5",
            "jardas_passe_por_jogo_ultimas_3", "jardas_passe_por_jogo_ultimas_5",
            "explosive_ultimas_3", "explosive_ultimas_5",
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
    p = p.dropna(subset=["week", "posteam", "defteam"])
    p["week"] = p["week"].astype(int)

    for col in ("qb_dropback", "qb_hit", "sack", "interception", "pass_td"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0).astype(int)
    for col in ("epa", "air_yards", "receiving_yards", "yards_gained"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    p["is_dropback"] = p.apply(
        lambda r: str(r.get("qb_dropback", "0")) in ("1", "1.0"), axis=1)
    p["is_pass_play"] = p.apply(
        lambda r: str(r.get("play_type", "")) == "pass", axis=1)
    p["is_explosive_pass"] = (
        (p["is_pass_play"] == 1) &
        (p["yards_gained"].fillna(0) >= 20)
    ).astype(int)

    # Para cada time, construir features de ofensa (proteção) e defesa (pressão/secundária).
    rows = []
    teams = sorted(set(p["posteam"].unique()) | set(p["defteam"].unique()))
    for team in teams:
        # --- OFENSA: proteção do QB do time (quando o time ataca) ---
        off = p[p["posteam"] == team]
        for week in range(1, max_week + 1):
            prior = off[off["week"] < week]
            if prior.empty:
                rows.append({
                    "team": team, "season": season, "week": week, "lado": "ofensa",
                    "pressao_permitida_ultimas_3": None,
                    "pressao_permitida_ultimas_5": None,
                    "sack_rate_ultimas_3": None,
                    "sack_rate_ultimas_5": None,
                    "epa_passe_ultimas_3": None,
                    "epa_passe_ultimas_5": None,
                })
                continue
            feats = {"team": team, "season": season, "week": week, "lado": "ofensa"}
            for wn, col in [(3, "pressao_permitida_ultimas_3"),
                            (5, "pressao_permitida_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db):
                    press = int(((db["qb_hit"] == 1) | (db["sack"] == 1)).sum())
                    feats[col] = round(press / len(db), 4)
                else:
                    feats[col] = None
            for wn, col in [(3, "sack_rate_ultimas_3"),
                            (5, "sack_rate_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db):
                    sacks = int((db["sack"] == 1).sum())
                    feats[col] = round(sacks / len(db), 4)
                else:
                    feats[col] = None
            for wn, col in [(3, "epa_passe_ultimas_3"),
                            (5, "epa_passe_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db) and "epa" in db.columns and db["epa"].notna().any():
                    feats[col] = round(float(db["epa"].mean()), 4)
                else:
                    feats[col] = None
            rows.append(feats)

        # --- DEFESA: pressão produzida e secundária (quando o time defende) ---
        deff = p[p["defteam"] == team]
        for week in range(1, max_week + 1):
            prior = deff[deff["week"] < week]
            if prior.empty:
                rows.append({
                    "team": team, "season": season, "week": week, "lado": "defesa",
                    "pressao_permitida_ultimas_3": None,
                    "pressao_permitida_ultimas_5": None,
                    "sack_rate_ultimas_3": None,
                    "sack_rate_ultimas_5": None,
                    "epa_passe_ultimas_3": None,
                    "epa_passe_ultimas_5": None,
                })
                continue
            feats = {"team": team, "season": season, "week": week, "lado": "defesa"}
            for wn, col in [(3, "pressao_permitida_ultimas_3"),
                            (5, "pressao_permitida_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db):
                    press = int(((db["qb_hit"] == 1) | (db["sack"] == 1)).sum())
                    feats[col] = round(press / len(db), 4)
                else:
                    feats[col] = None
            for wn, col in [(3, "sack_rate_ultimas_3"),
                            (5, "sack_rate_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db):
                    sacks = int((db["sack"] == 1).sum())
                    feats[col] = round(sacks / len(db), 4)
                else:
                    feats[col] = None
            for wn, col in [(3, "epa_passe_ultimas_3"),
                            (5, "epa_passe_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                db = recent[recent["is_dropback"] == 1]
                if len(db) and "epa" in db.columns and db["epa"].notna().any():
                    feats[col] = round(float(db["epa"].mean()), 4)
                else:
                    feats[col] = None
            # EPA contra passe (defesa): EPA médio em jogadas de passe sofridas.
            for wn, col in [(3, "epa_contra_passe_ultimas_3"),
                            (5, "epa_contra_passe_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                pp = recent[recent["is_pass_play"] == 1]
                if len(pp) and "epa" in pp.columns and pp["epa"].notna().any():
                    feats[col] = round(float(pp["epa"].mean()), 4)
                else:
                    feats[col] = None
            # Jardas aéreas por jogo (defesa).
            for wn, col in [(3, "jardas_passe_por_jogo_ultimas_3"),
                            (5, "jardas_passe_por_jogo_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                pp = recent[recent["is_pass_play"] == 1]
                if len(pp):
                    jogos = pp["game_id"].nunique()
                    yards = pp["receiving_yards"].sum() if "receiving_yards" in pp.columns else 0.0
                    feats[col] = round(float(yards) / max(jogos, 1), 4)
                else:
                    feats[col] = None
            # Explosive plays (>=20 jardas de passe) permitidas.
            for wn, col in [(3, "explosive_ultimas_3"),
                            (5, "explosive_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                feats[col] = int((recent["is_explosive_pass"] == 1).sum()) \
                    if len(recent) > 0 else None
            rows.append(feats)

    return pd.DataFrame(rows)
