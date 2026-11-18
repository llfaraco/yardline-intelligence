"""Configuração canônica do Yardline Intelligence v0.4.1.

Tudo o que varia de ambiente (caminhos, temporadas, semanas, scoring) vive aqui.
Nenhum dado é inventado: os valores default apontam para fontes reais do nflverse.
"""

from pathlib import Path

# Diretório raiz do motor (isolado do frontend/backend Node).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PARQUET_DIR = DATA_DIR / "parquet"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
MODELS_DIR = DATA_DIR / "models"

# Temporada corrente e janela de treino/validação.
CURRENT_SEASON = 2026
TRAIN_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023]
VALIDATE_SEASON = 2024
TEST_SEASON = 2025

# Scoring Half-PPR (preset oficial da Yardline, idêntico ao servidor).
HALF_PPR = {
    "pass_yd": 0.04, "pass_td": 4, "pass_int": -1, "pass_2pt": 2,
    "rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2,
    "rec": 0.5, "rec_yd": 0.1, "rec_td": 6, "rec_2pt": 2,
    "fum_lost": -2,
}

# Fontes autorizadas (nflverse / DynastyProcess / ffverse).
SOURCES = {
    "pbp": "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz",
    "snaps": "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv",
    "injuries": "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv",
    "ids": "https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv",
    "games": "https://github.com/nflverse/nfldata/raw/master/data/games.csv",
}

# Posições com valor fantasy (o resto é filtrado).
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}

# Identificadores canônicos de WR usados nos primeiros testes de backtest.
WR_SEED_2026 = {
    "Ja'Marr Chase": "7564",
    "Justin Jefferson": "6794",
    "CeeDee Lamb": "6796",
}

for _d in (CACHE_DIR, PARQUET_DIR, EXPERIMENTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
