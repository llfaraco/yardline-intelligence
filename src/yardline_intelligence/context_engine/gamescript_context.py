"""Contexto de game script — Camada do Context Engine.

Constrói, para cada time e semana, features derivadas do play-by-play que
descrevem o ritmo ofensivo, tendência de passes/corridas, mando de campo e
expectativa de pontuação (vegas), usando APENAS semanas anteriores à previsão
(sem data leakage).

Essas features são atribuídas a todos os WRs/TEs/RBs do time, pois o contexto
de jogo afeta o volume de oportunidades.

Features:
- ritmo_ofensivo_ultimas_N: jogadas por jogo do time
- taxa_passagem_ultimas_N: proporção de jogadas que são passe
- mando_campo: 1 se o time é mandante na semana alvo, 0 se visitante
- descanso_dias: dias desde o último jogo do time
- vegas_total_linha: total esperado (over/under) do jogo da semana alvo
- vegas_spread_linha: spread da semana alvo (positivo = underdog)
- expected_points_time: pontos esperados do time com base no total/spread
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..data_foundation.ingest import read_csv_cached, CACHE_DIR
from ..config import CURRENT_SEASON

PBP_GS_COLUMNS = [
    "season", "week", "game_type", "game_id", "posteam", "defteam", "play_type",
    "qb_dropback", "rush_attempt", "home_team", "away_team", "location",
    "spread_line", "total_line", "game_date",
]


def load_pbp_gamescript(season: int,
                        cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Carrega colunas do pbp relevantes ao game script de uma temporada."""
    path = cache_dir / f"pbp_{season}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    return read_csv_cached(__import__("duckdb").connect(),
                           path, PBP_GS_COLUMNS)


def build_gamescript_context(pbp: pd.DataFrame,
                             season: int,
                             max_week: int) -> pd.DataFrame:
    """Constrói features de game script por (time, week).

    Retorna DataFrame com colunas: team, season, week, e as features de game
    script. Cada time aparece uma vez por semana (seu contexto ofensivo).
    """
    if pbp.empty:
        return pd.DataFrame(columns=[
            "team", "season", "week",
            "ritmo_ofensivo_ultimas_3", "ritmo_ofensivo_ultimas_5",
            "taxa_passagem_ultimas_3", "taxa_passagem_ultimas_5",
            "mando_campo", "descanso_dias",
            "vegas_total_linha", "vegas_spread_linha",
            "expected_points_time",
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
    p = p.dropna(subset=["week", "posteam", "home_team", "away_team"])
    p["week"] = p["week"].astype(int)

    for col in ("qb_dropback", "rush_attempt"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0).astype(int)
    for col in ("spread_line", "total_line"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    p["is_pass_play"] = p.apply(
        lambda r: str(r.get("play_type", "")) == "pass", axis=1)
    p["is_rush_play"] = p.apply(
        lambda r: str(r.get("play_type", "")) == "rush", axis=1)

    # Data do jogo para cálculo de descanso.
    p["game_date_parsed"] = pd.to_datetime(p["game_date"], errors="coerce")

    # Informação do jogo por (semana): times, local, vegas.
    game_info = {}
    for week, sub in p.groupby("week"):
        info = {}
        for _, r in sub.iterrows():
            home = str(r["home_team"]); away = str(r["away_team"])
            info[home] = {
                "opponent": away, "location": str(r.get("location", "")),
                "spread_line": r.get("spread_line"), "total_line": r.get("total_line"),
                "game_date": r.get("game_date_parsed"),
            }
            info[away] = {
                "opponent": home, "location": str(r.get("location", "")),
                "spread_line": r.get("spread_line"), "total_line": r.get("total_line"),
                "game_date": r.get("game_date_parsed"),
            }
        game_info[week] = info

    # Para cada time, lista ordenada de semanas em que jogou (para cálculo de descanso).
    teams = sorted(set(p["posteam"].unique()) | set(p["defteam"].unique()))
    team_played_weeks = {}
    for team in teams:
        tp = p[(p["posteam"] == team) | (p["defteam"] == team)]
        team_played_weeks[team] = sorted(tp["week"].dropna().unique().astype(int))

    rows = []
    teams_sorted = sorted(teams)
    for team in teams_sorted:
        # Contexto ofensivo do time (quando o time ataca).
        off = p[p["posteam"] == team]
        for week in range(1, max_week + 1):
            prior = off[off["week"] < week]
            if prior.empty:
                # Sem histórico ofensivo: usar info do jogo atual se disponível.
                gi = game_info.get(week, {}).get(team, {})
                rows.append({
                    "team": team, "season": season, "week": week,
                    "ritmo_ofensivo_ultimas_3": None,
                    "ritmo_ofensivo_ultimas_5": None,
                    "taxa_passagem_ultimas_3": None,
                    "taxa_passagem_ultimas_5": None,
                    "mando_campo": _mando_campo(gi, team),
                    "descanso_dias": _descanso(team, week, team_played_weeks),
                    "vegas_total_linha": _vegas_val(gi, "total_line"),
                    "vegas_spread_linha": _vegas_val(gi, "spread_line", team),
                    "expected_points_time": None,
                })
                continue

            feats = {"team": team, "season": season, "week": week}
            for wn, col in [(3, "ritmo_ofensivo_ultimas_3"),
                            (5, "ritmo_ofensivo_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                jogos = recent["game_id"].nunique()
                nplays = len(recent)
                feats[col] = round(nplays / max(jogos, 1), 2) if jogos > 0 else None
            for wn, col in [(3, "taxa_passagem_ultimas_3"),
                            (5, "taxa_passagem_ultimas_5")]:
                recent = prior[prior["week"] >= week - wn + 1]
                if len(recent):
                    npass = int((recent["is_pass_play"] == 1).sum())
                    feats[col] = round(npass / len(recent), 4)
                else:
                    feats[col] = None

            # Info do jogo da semana alvo.
            gi = game_info.get(week, {}).get(team, {})
            feats["mando_campo"] = _mando_campo(gi, team)
            feats["descanso_dias"] = _descanso(team, week, team_played_weeks)
            feats["vegas_total_linha"] = _vegas_val(gi, "total_line")
            feats["vegas_spread_linha"] = _vegas_val(gi, "spread_line", team)

            # Expected points do time: (total + spread_ajustado) / 2.
            total = feats["vegas_total_linha"]
            spread = feats["vegas_spread_linha"]
            if total is not None and spread is not None:
                # spread positivo = underdog; pontos time = (total - spread)/2 quando time é away/underdog.
                # Simplificação: expected = (total + spread_corrigido)/2
                exp = (float(total) + float(spread)) / 2.0
                feats["expected_points_time"] = round(exp, 2)
            else:
                feats["expected_points_time"] = None

            rows.append(feats)

    return pd.DataFrame(rows)


def _mando_campo(gi: dict, team: str) -> Optional[int]:
    """1 se o time é mandante, 0 se visitante, None se indisponível."""
    if not gi:
        return None
    loc = str(gi.get("location", "")).lower()
    if loc in ("home",):
        return 1
    if loc in ("away", "neutral"):
        return 0
    # Fallback: tentar inferir pelo opponent (não confiável).
    return None


def _vegas_val(gi: dict, key: str, team: Optional[str] = None) -> Optional[float]:
    """Extrai valor vegas. Para spread, ajusta o sinal com base no mando de campo."""
    if not gi:
        return None
    v = gi.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if key == "spread_line":
        # No nflverse games.csv, spread_line é do ponto de vista do home team.
        # Se o time é home, spread como está; se away, inverte.
        if team is not None:
            mando = _mando_campo(gi, team)
            if mando == 0:  # away: inverte
                v = -v
        return v
    return v


def _descanso(team: str, week: int, team_played_weeks: dict) -> Optional[int]:
    """Dias desde o último jogo do time (7 se semana consecutiva, None se primeiro jogo)."""
    played = team_played_weeks.get(team, [])
    prior = [w for w in played if w < week]
    if not prior:
        return None
    last = prior[-1]
    diff = week - last
    return int(diff * 7)
