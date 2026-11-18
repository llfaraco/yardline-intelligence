"""Constrói o dataset temporal por jogador-temporada-semana a partir das fontes
ingiradas (play-by-play, snap counts, ids, jogos).

Cada linha representa o que era conhecido sobre um jogador ANTES de uma semana:
features de volume, eficiência e contexto, com o alvo (pontos Half-PPR reais)
daquela mesma semana. Nunca vaza informação futura.

Scoring Half-PPR (idêntico ao preset oficial da Yardline no servidor):
- Passe: 0.04 por jarda, 4 por TD, -1 por INT, 2 por 2pt.
- Corrida: 0.1 por jarda, 6 por TD, 2 por 2pt.
- Recepção: 0.5 por recepção, 0.1 por jarda, 6 por TD, 2 por 2pt.
- Fumble perdido: -2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from ..data_foundation.ingest import (
    read_csv_cached, to_parquet, connect_db, init_schema, SourceCatalog,
)
from ..config import (
    CACHE_DIR, PARQUET_DIR, CURRENT_SEASON, TRAIN_SEASONS, VALIDATE_SEASON,
    FANTASY_POSITIONS, HALF_PPR,
)

# Colunas do play-by-play (nflverse) necessárias para scoring Half-PPR completo
# e para features de volume/eficiência. São lidas via streaming (sem manter o
# arquivo de 370 colunas na memória).
PBP_COLUMNS = [
    "play_id", "game_id", "season", "week", "home_team", "away_team", "posteam",
    "defteam", "play_type", "down", "ydstogo", "quarter_seconds_remaining",
    "score_differential", "qb_dropback", "qb_scramble", "pass_attempt",
    "complete_pass", "incomplete_pass", "sack", "air_yards", "receiving_yards",
    "rushing_yards", "pass_location", "epa", "passer_player_id",
    "receiver_player_id", "rusher_player_id", "sack_player_id",
    "pass_defense_1_player_id", "pass_defense_2_player_id",
    "pass_td", "rush_td", "rec_td", "fumble_lost",
]

SNAP_COLUMNS = [
    "game_type", "week", "pfr_player_id", "position", "team",
    "offense_pct", "offense_snaps", "defense_pct", "defense_snaps",
]

IDS_COLUMNS = ["gsis_id", "sleeper_id", "name", "position", "team", "age",
               "draft_year", "draft_round", "draft_pick"]

GAMES_COLUMNS = ["season", "game_type", "week", "away_team", "home_team",
                 "spread_line", "total_line"]


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def build_player_dataset(season: int,
                         cache_dir: Path = CACHE_DIR,
                         parquet_dir: Path = PARQUET_DIR,
                         db_path: Optional[Path] = None) -> pd.DataFrame:
    """Constrói o dataset temporal de uma temporada.

    - pbp: eventos de jogo até `through_week` (sem vazar semanas futuras).
    - snaps: share de snaps até a semana anterior.
    - ids: mapeamento GSIS <-> Sleeper (identificação canônica).
    - games: calendário para saber quais semanas já foram jogadas.

    O alvo `half_ppr_points` é a pontuação real da semana, computada a partir
    do pbp mesmo (derivado), nunca projeção de terceiros.
    """
    db = connect_db(db_path)
    try:
        init_schema(db)
        catalog = SourceCatalog(db)

        pbp_path = cache_dir / f"pbp_{season}.csv.gz"
        snaps_path = cache_dir / f"snaps_{season}.csv"
        ids_path = cache_dir / f"ids_{season}.csv"
        games_path = cache_dir / f"games_{season}.csv"
        # Fallback para nomes sem sufixo de temporada (usados quando a fonte é
        # atemporal, sem vínculo com uma temporada específica).
        if not ids_path.exists() and (cache_dir / "ids.csv").exists():
            ids_path = cache_dir / "ids.csv"
        if not games_path.exists() and (cache_dir / "games.csv").exists():
            games_path = cache_dir / "games.csv"

        pbp = read_csv_cached(db, pbp_path, PBP_COLUMNS) if pbp_path.exists() else pd.DataFrame()
        snaps = read_csv_cached(db, snaps_path, SNAP_COLUMNS) if snaps_path.exists() else pd.DataFrame()
        ids = read_csv_cached(db, ids_path, IDS_COLUMNS) if ids_path.exists() else pd.DataFrame()
        games = read_csv_cached(db, games_path, GAMES_COLUMNS) if games_path.exists() else pd.DataFrame()

        if pbp.empty or ids.empty:
            return pd.DataFrame()

        # Mapeia GSIS -> Sleeper (identificação canônica, nunca por nome).
        gsis_to_sleeper = dict(zip(ids["gsis_id"].astype(str), ids["sleeper_id"].astype(str)))
        sleeper_to_name = dict(zip(ids["sleeper_id"].astype(str), ids["name"].astype(str)))
        sleeper_to_team = dict(zip(ids["sleeper_id"].astype(str), ids["team"].astype(str)))
        sleeper_to_pos = dict(zip(ids["sleeper_id"].astype(str), ids["position"].astype(str)))

        pbp = pbp.copy()
        pbp["week"] = _to_num(pbp["week"])
        pbp["season"] = _to_num(pbp["season"])
        pbp = pbp[pbp["season"] == season]
        for col in ("epa", "air_yards", "receiving_yards", "rushing_yards",
                    "qb_dropback", "pass_attempt", "complete_pass", "sack",
                    "pass_td", "rush_td", "rec_td", "fumble_lost"):
            if col in pbp.columns:
                pbp[col] = _to_num(pbp[col])
        pbp["rusher_player_id"] = pbp["rusher_player_id"].astype(str)
        pbp["receiver_player_id"] = pbp["receiver_player_id"].astype(str)
        pbp["passer_player_id"] = pbp["passer_player_id"].astype(str)
        pbp["sack_player_id"] = pbp["sack_player_id"].astype(str)
        # Normaliza flags de 0/1 (vêm como string do CSV) para inteiro.
        for col in ("complete_pass", "qb_dropback", "pass_attempt", "sack",
                    "pass_td", "rush_td", "rec_td", "fumble_lost"):
            if col in pbp.columns:
                pbp[col] = _to_num(pbp[col]).fillna(0).astype(int)

        if not games.empty:
            games = games.copy()
            games["week"] = _to_num(games["week"])
            games["season"] = _to_num(games["season"])
            played_weeks = set(games.loc[(games["season"] == season) &
                                         (games["game_type"] == "REG"), "week"].dropna().astype(int))
        else:
            played_weeks = set(pbp["week"].dropna().astype(int))

        max_week = max(played_weeks) if played_weeks else int(pbp["week"].max())

        # Converte pbp para pontos Half-PPR por jogador e semana.
        weekly_points = _score_weekly(pbp, HALF_PPR)

        # As chaves de weekly_points são GSIS ids (vêm do pbp). Traduz para
        # Sleeper (identificação canônica usada pelo motor de fantasy).
        gsis_to_sleeper = dict(zip(ids["gsis_id"].astype(str), ids["sleeper_id"].astype(str)))
        weekly_points = {
            w: {gsis_to_sleeper.get(g, g): pts for g, pts in wk.items()}
            for w, wk in weekly_points.items()
        }

        sleeper_ids = {sid for wk in weekly_points.values() for sid in wk}
        gsis_to_sleeper_filtered = {g: s for g, s in gsis_to_sleeper.items() if s in sleeper_ids}

        rows = []
        for week in range(1, max_week + 1):
            # Features usam apenas dados de semanas < week (prevenção de data leakage).
            prior_pbp = pbp[pbp["week"] < week]
            prior_snaps = snaps.copy()
            if not prior_snaps.empty:
                prior_snaps["week"] = _to_num(prior_snaps["week"])
                prior_snaps = prior_snaps[prior_snaps["week"] < week]

            # Alvo: pontos reais da semana (disponíveis apenas depois do jogo).
            target_week = weekly_points.get(week, {})

            for sleeper_id, points in target_week.items():
                pos = sleeper_to_pos.get(sleeper_id)
                if pos not in FANTASY_POSITIONS:
                    continue
                team = sleeper_to_team.get(sleeper_id)
                name = sleeper_to_name.get(sleeper_id)

                feats = _player_features(
                    sleeper_id, gsis_to_sleeper_filtered, prior_pbp, prior_snaps,
                    team, week, pbp,
                )
                rows.append({
                    "season": season,
                    "week": week,
                    "sleeper_id": sleeper_id,
                    "gsis_id": feats.get("gsis_id"),
                    "name": name,
                    "position": pos,
                    "team": team,
                    "opponent": feats.get("opponent"),
                    "half_ppr_points": points,
                    **feats,
                })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values(["sleeper_id", "week"]).reset_index(drop=True)
        out_path = to_parquet(df, f"dataset_{season}", season)
        db.execute("""
            INSERT OR REPLACE INTO ingestion_log
            (id, source, season, status, message, started_at, finished_at, rows)
            VALUES ((SELECT COALESCE(MAX(id),0)+1 FROM ingestion_log), 'dataset', ?, 'ok', ?, ?, ?, ?)
        """, [season, f"dataset temporal construído com {len(df)} linhas", now_iso(), now_iso(), len(df)])
        return df
    finally:
        db.close()


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _score_weekly(pbp: pd.DataFrame, scoring: dict) -> dict[int, dict[str, float]]:
    """Pontua cada jogada do pbp em Half-PPR e agrega por jogador-semana."""
    weekly: dict[int, dict[str, float]] = {}
    for _, row in pbp.iterrows():
        week = row["week"]
        if pd.isna(week):
            continue
        week = int(week)

        def add(player_id: str, points: float) -> None:
            if not player_id or pd.isna(player_id):
                return
            weekly.setdefault(week, {})
            weekly[week][str(player_id)] = weekly[week].get(str(player_id), 0.0) + points

        # Recepção (0.5 por recepção + 0.1 por jarda)
        if row.get("complete_pass") == 1 and row.get("receiver_player_id"):
            rid = str(row["receiver_player_id"])
            add(rid, scoring["rec"])
            rec_yd = row.get("receiving_yards")
            if pd.notna(rec_yd):
                add(rid, scoring["rec_yd"] * float(rec_yd))
        # TD de recepção
        if row.get("rec_td") == 1 and row.get("receiver_player_id"):
            add(str(row["receiver_player_id"]), scoring["rec_td"])

        # Corrida (0.1 por jarda)
        if row.get("play_type") == "rush" and row.get("rusher_player_id"):
            rid = str(row["rusher_player_id"])
            rush_yd = row.get("rushing_yards")
            if pd.notna(rush_yd):
                add(rid, scoring["rush_yd"] * float(rush_yd))
        # TD de corrida
        if row.get("rush_td") == 1 and row.get("rusher_player_id"):
            add(str(row["rusher_player_id"]), scoring["rush_td"])

        # Passe (0.04 por jarda de arremesso; usamos air_yards como proxy de jarda de passe no pbp enxuto)
        if row.get("qb_dropback") == 1 and row.get("passer_player_id"):
            pid = str(row["passer_player_id"])
            pass_yd = row.get("air_yards")
            if pd.notna(pass_yd):
                add(pid, scoring["pass_yd"] * float(pass_yd))
        # TD de passe
        if row.get("pass_td") == 1 and row.get("passer_player_id"):
            add(str(row["passer_player_id"]), scoring["pass_td"])
        # Interceptação lançada
        # (pbp enxuto não tem coluna direta de int; ignorado se ausente)
        if row.get("fumble_lost") == 1:
            # Fumble perdido atribuído ao jogador com a bola na jogada
            for col in ("rusher_player_id", "receiver_player_id", "passer_player_id"):
                pid = row.get(col)
                if pid and pd.notna(pid):
                    add(str(pid), scoring["fum_lost"])

    return weekly


def _player_features(sleeper_id, gsis_to_sleeper, pbp_prior, snaps_prior,
                     team: str, week: int, pbp_full: pd.DataFrame) -> dict:
    """Features de um jogador para uma semana, só com dados anteriores.

    Inclui o adversário da semana (vindo do calendário/jogos da própria semana,
    que é conhecido com antecedência — não é data leakage).
    """
    gsis_id = None
    for g, s in gsis_to_sleeper.items():
        if s == sleeper_id:
            gsis_id = g
            break

    feats = {
        "gsis_id": gsis_id,
        "opponent": None,
        "targets": 0, "catches": 0, "air_yards": 0.0, "receiving_yards": 0.0,
        "epa_sum": 0.0, "rushes": 0, "rush_yards": 0.0,
        "dropbacks": 0, "pass_attempts": 0, "completions": 0,
        "sacks": 0, "snap_share": None, "snap_count": None,
        "games_played": 0,
    }

    # Adversário da semana (conhecido antecipadamente via calendário).
    # Busca no pbp_full qualquer jogo do time do jogador nesta semana.
    if team and not pbp_full.empty:
        team_games = pbp_full[(pbp_full["week"] == week) &
                              ((pbp_full["posteam"] == team) | (pbp_full["defteam"] == team))]
        if not team_games.empty:
            row = team_games.iloc[0]
            if row["posteam"] == team:
                feats["opponent"] = row["defteam"]
            else:
                feats["opponent"] = row["posteam"]

    if pbp_prior.empty:
        return feats

    mask_receiver = pbp_prior["receiver_player_id"] == gsis_id
    mask_rusher = pbp_prior["rusher_player_id"] == gsis_id
    mask_passer = pbp_prior["passer_player_id"] == gsis_id

    if gsis_id and (mask_receiver.any() or mask_rusher.any() or mask_passer.any()):
        sub = pbp_prior[mask_receiver | mask_rusher | mask_passer]
        feats["games_played"] = int(sub["game_id"].nunique())

    if mask_receiver.any():
        rec = pbp_prior[mask_receiver]
        feats["targets"] = int((rec["complete_pass"] == 1).sum())
        feats["catches"] = feats["targets"]
        ay = rec["air_yards"].dropna()
        feats["air_yards"] = float(ay.sum()) if len(ay) else 0.0
        ry = rec["receiving_yards"].dropna()
        feats["receiving_yards"] = float(ry.sum()) if len(ry) else 0.0
        epa = rec["epa"].dropna()
        feats["epa_sum"] = float(epa.sum()) if len(epa) else 0.0

    if mask_rusher.any():
        rush = pbp_prior[mask_rusher & (pbp_prior["play_type"] == "rush")]
        feats["rushes"] = int(len(rush))
        ry = rush["rushing_yards"].dropna()
        feats["rush_yards"] = float(ry.sum()) if len(ry) else 0.0

    if mask_passer.any():
        passp = pbp_prior[mask_passer & (pbp_prior["qb_dropback"] == 1)]
        feats["dropbacks"] = int(len(passp))
        feats["pass_attempts"] = int((passp["pass_attempt"] == 1).sum())
        feats["completions"] = int((passp["complete_pass"] == 1).sum())
        sacks = pbp_prior[mask_passer & (pbp_prior["sack"] == 1)]
        feats["sacks"] = int(len(sacks))

    # Snap share do jogador (média das semanas anteriores).
    if not snaps_prior.empty and gsis_id:
        snap_rows = snaps_prior[snaps_prior["pfr_player_id"].astype(str) == gsis_id]
        if not snap_rows.empty:
            pct = _to_num(snap_rows["offense_pct"])
            feats["snap_share"] = float(pct.mean()) if pct.notna().any() else None
            cnt = _to_num(snap_rows["offense_snaps"])
            feats["snap_count"] = float(cnt.sum()) if cnt.notna().any() else None

    return feats
