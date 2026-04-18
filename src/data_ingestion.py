import os
import time
import logging
import pandas as pd
from requests.exceptions import RequestException
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from nba_api.stats.endpoints import (
    teamgamelogs,
    playergamelogs,
    leaguedashplayerstats,
    leaguedashteamstats,
)
from nba_api.stats.static import teams, players


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

SEASON      = "2025-26"
SEASON_TYPE = "Regular Season"
DELAY       = 2  # segundos entre llamadas a la API
NBA_TIMEOUT = 60
NBA_MAX_RETRIES = 4



def get_engine():
    user = os.getenv("POSTGRES_USER", "nba_user")
    pw   = os.getenv("POSTGRES_PASSWORD", "nba_pass")
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5433")
    db   = os.getenv("POSTGRES_DB", "nba_database")

    url = f"postgresql+psycopg://{user}:{pw}@{host}:{port}/{db}"

    return create_engine(
        url,
        connect_args={"client_encoding": "utf8"}
    )


def fetch_nba_dataframe(endpoint_cls, **params):
    endpoint_name = endpoint_cls.__name__

    for attempt in range(1, NBA_MAX_RETRIES + 1):
        try:
            time.sleep(DELAY)
            endpoint = endpoint_cls(
                timeout=NBA_TIMEOUT,
                **params
            )
            return endpoint.get_data_frames()[0]
        except (RequestException, ValueError, KeyError) as exc:
            if attempt == NBA_MAX_RETRIES:
                logger.exception(
                    "%s fallo tras %s intentos. stats.nba.com cerro o devolvio "
                    "una respuesta invalida.",
                    endpoint_name,
                    NBA_MAX_RETRIES
                )
                raise

            wait_seconds = min(45, DELAY * (2 ** attempt))
            logger.warning(
                "%s fallo en intento %s/%s: %s. Reintentando en %ss...",
                endpoint_name,
                attempt,
                NBA_MAX_RETRIES,
                exc,
                wait_seconds
            )
            time.sleep(wait_seconds)



def create_schema(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS dim_teams (
        team_id      BIGINT PRIMARY KEY,
        full_name    TEXT,
        abbreviation TEXT,
        nickname     TEXT,
        city         TEXT,
        state        TEXT
    );

    CREATE TABLE IF NOT EXISTS dim_players (
        player_id BIGINT PRIMARY KEY,
        full_name TEXT,
        is_active BOOLEAN
    );

    CREATE TABLE IF NOT EXISTS fact_team_game_logs (
        game_id    TEXT,
        team_id    BIGINT,
        game_date  DATE,
        matchup    TEXT,
        wl         TEXT,
        pts        INT,
        reb        INT,
        ast        INT,
        stl        INT,
        blk        INT,
        tov        INT,
        fg_pct     FLOAT,
        fg3_pct    FLOAT,
        ft_pct     FLOAT,
        plus_minus FLOAT,
        PRIMARY KEY (game_id, team_id)
    );

    CREATE TABLE IF NOT EXISTS fact_player_game_logs (
        game_id    TEXT,
        player_id  BIGINT,
        team_id    BIGINT,
        game_date  DATE,
        matchup    TEXT,
        wl         TEXT,
        min        TEXT,
        pts        INT,
        reb        INT,
        ast        INT,
        stl        INT,
        blk        INT,
        tov        INT,
        fg_pct     FLOAT,
        fg3_pct    FLOAT,
        ft_pct     FLOAT,
        plus_minus FLOAT,
        PRIMARY KEY (game_id, player_id)
    );

    CREATE TABLE IF NOT EXISTS fact_player_season_stats (
        player_id   BIGINT,
        player_name TEXT,
        team_abbrev TEXT,
        gp          INT,
        pts         FLOAT,
        reb         FLOAT,
        ast         FLOAT,
        stl         FLOAT,
        blk         FLOAT,
        tov         FLOAT,
        fg_pct      FLOAT,
        fg3_pct     FLOAT,
        ft_pct      FLOAT,
        ts_pct      FLOAT,
        season      TEXT,
        PRIMARY KEY (player_id, season)
    );

    CREATE TABLE IF NOT EXISTS fact_team_season_stats (
        team_id     BIGINT,
        team_name   TEXT,
        team_abbrev TEXT,
        gp          INT,
        w           INT,
        l           INT,
        w_pct       FLOAT,
        pts         FLOAT,
        reb         FLOAT,
        ast         FLOAT,
        stl         FLOAT,
        blk         FLOAT,
        tov         FLOAT,
        fg_pct      FLOAT,
        fg3_pct     FLOAT,
        ft_pct      FLOAT,
        season      TEXT,
        PRIMARY KEY (team_id, season)
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    logger.info("Schema verificado correctamente.")


def ingest_teams(engine):
    logger.info("Ingresando equipos NBA...")
    df = pd.DataFrame(teams.get_teams())
    df = df.rename(columns={"id": "team_id"})
    df[["team_id", "full_name", "abbreviation",
        "nickname", "city", "state"]].to_sql(
        "dim_teams", engine, if_exists="replace", index=False
    )
    logger.info(f"  {len(df)} equipos guardados.")


def ingest_players(engine):
    logger.info("Ingresando jugadores activos NBA...")
    df = pd.DataFrame(players.get_active_players())
    df = df.rename(columns={"id": "player_id"})
    df[["player_id", "full_name", "is_active"]].to_sql(
        "dim_players", engine, if_exists="replace", index=False
    )
    logger.info(f"  {len(df)} jugadores guardados.")


def ingest_team_game_logs(engine):
    logger.info("Ingresando game logs de equipos (temporada 2025-26)...")
    df = fetch_nba_dataframe(
        teamgamelogs.TeamGameLogs,
        season_nullable=SEASON,
        season_type_nullable=SEASON_TYPE
    )
    df.columns = df.columns.str.lower()
    df["game_date"] = pd.to_datetime(df["game_date"])
    cols = ["game_id", "team_id", "game_date", "matchup", "wl",
            "pts", "reb", "ast", "stl", "blk", "tov",
            "fg_pct", "fg3_pct", "ft_pct", "plus_minus"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_sql("fact_team_game_logs", engine,
              if_exists="replace", index=False)
    logger.info(f"  {len(df)} registros de equipos guardados.")


def ingest_player_game_logs(engine):
    logger.info("Ingresando game logs de jugadores (puede tardar 1-2 min)...")
    df = fetch_nba_dataframe(
        playergamelogs.PlayerGameLogs,
        season_nullable=SEASON,
        season_type_nullable=SEASON_TYPE
    )
    df.columns = df.columns.str.lower()
    df["game_date"] = pd.to_datetime(df["game_date"])
    cols = ["game_id", "player_id", "team_id", "game_date", "matchup", "wl",
            "min", "pts", "reb", "ast", "stl", "blk", "tov",
            "fg_pct", "fg3_pct", "ft_pct", "plus_minus"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_sql("fact_player_game_logs", engine,
              if_exists="replace", index=False)
    logger.info(f"  {len(df)} registros de jugadores guardados.")


def ingest_player_season_stats(engine):
    logger.info("Ingresando estadísticas de temporada por jugador...")
    df = fetch_nba_dataframe(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        per_mode_detailed="PerGame"
    )
    df.columns = df.columns.str.lower()
    df["season"] = SEASON
    df = df.rename(columns={"team_abbreviation": "team_abbrev"})
    cols = ["player_id", "player_name", "team_abbrev", "gp",
            "pts", "reb", "ast", "stl", "blk", "tov",
            "fg_pct", "fg3_pct", "ft_pct", "season"]
    df = df[cols]
    df.to_sql("fact_player_season_stats", engine,
              if_exists="replace", index=False)
    logger.info(f"  {len(df)} jugadores con stats de temporada guardados.")


def ingest_team_season_stats(engine):
    logger.info("Ingresando estadísticas de temporada por equipo...")
    df = fetch_nba_dataframe(
        leaguedashteamstats.LeagueDashTeamStats,
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        per_mode_detailed="PerGame"
    )
    df.columns = df.columns.str.lower()
    df["season"] = SEASON
    df = df.rename(columns={
        "team_name": "team_name",
        "team_abbreviation": "team_abbrev"
    })
    cols = ["team_id", "team_name", "team_abbrev", "gp", "w", "l", "w_pct",
            "pts", "reb", "ast", "stl", "blk", "tov",
            "fg_pct", "fg3_pct", "ft_pct", "season"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_sql("fact_team_season_stats", engine,
              if_exists="replace", index=False)
    logger.info(f"  {len(df)} equipos con stats de temporada guardados.")


def run_ingestion():
    logger.info("=" * 50)
    logger.info("INICIO INGESTA NBA 2025-26")
    logger.info("=" * 50)
    engine = get_engine()
    create_schema(engine)
    ingest_teams(engine)
    ingest_players(engine)
    ingest_team_game_logs(engine)
    ingest_player_game_logs(engine)
    ingest_player_season_stats(engine)
    ingest_team_season_stats(engine)
    logger.info("=" * 50)
    logger.info("INGESTA COMPLETA - Datos listos en PostgreSQL")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_ingestion()
