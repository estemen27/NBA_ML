import os
import warnings
import pandas as pd
import psycopg2
from contextlib import contextmanager

_DB = dict(
    host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
    port=int(os.getenv("POSTGRES_PORT", 5433)),
    user=os.getenv("POSTGRES_USER", "nba_user"),
    password=os.getenv("POSTGRES_PASSWORD", "nba_pass"),
    dbname=os.getenv("POSTGRES_DB", "nba_database"),
)


@contextmanager
def get_conn():
    conn = psycopg2.connect(**_DB)
    try:
        yield conn
    finally:
        conn.close()


def query_df(sql: str, params=None) -> pd.DataFrame:
    with get_conn() as conn:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.read_sql(sql, conn, params=params)
