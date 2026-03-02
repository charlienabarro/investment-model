import sqlite3
from contextlib import contextmanager
from typing import Iterator
from .config import DB_PATH

@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS prices_daily (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (date, ticker)
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            score REAL,
            target_weight REAL,
            reasons TEXT,
            PRIMARY KEY (asof_date, ticker)
        );
        """)

        # Latest model holdings snapshot for an asof_date
        conn.execute("""
        CREATE TABLE IF NOT EXISTS model_holdings (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            target_weight REAL NOT NULL,
            price REAL NOT NULL,
            shares REAL NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (asof_date, ticker)
        );
        """)

        # Trade plan from previous snapshot to latest snapshot
        conn.execute("""
        CREATE TABLE IF NOT EXISTS model_trades (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            trade_action TEXT NOT NULL,
            shares_delta REAL NOT NULL,
            est_notional REAL NOT NULL,
            PRIMARY KEY (asof_date, ticker)
        );
        """)

        conn.execute("""
                CREATE TABLE IF NOT EXISTS features_daily (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    mom_12_1 REAL,
                    mom_6_1 REAL,
                    vol_63 REAL,
                    ma_200_ratio REAL,
                    maxdd_252 REAL,
                    PRIMARY KEY (ticker, date)
                )
                """)

def get_last_date_for_ticker(conn: sqlite3.Connection, ticker: str) -> str | None:
    cur = conn.execute(
        "SELECT MAX(date) FROM prices_daily WHERE ticker = ?",
        (ticker,)
    )
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else None