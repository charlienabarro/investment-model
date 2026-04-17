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

        conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            asof_date TEXT PRIMARY KEY,
            starting_equity REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshot_holdings (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL NOT NULL,
            weight REAL NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (asof_date, ticker)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_performance (
            asof_date TEXT PRIMARY KEY,
            end_date TEXT NOT NULL,
            start_equity REAL NOT NULL,
            end_equity REAL NOT NULL,
            pnl_gbp REAL NOT NULL,
            pnl_pct REAL NOT NULL,
            max_drawdown REAL NOT NULL,
            spy_start REAL,
            spy_end REAL,
            spy_pnl_pct REAL,
            created_at TEXT NOT NULL
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_contrib (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            pnl_gbp REAL NOT NULL,
            pnl_pct REAL NOT NULL,
            contribution_pct REAL NOT NULL,
            PRIMARY KEY (asof_date, ticker)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS learned_params (
            ticker TEXT PRIMARY KEY,
            score_adjustment REAL NOT NULL,
            last_signal REAL NOT NULL,
            effective_asof TEXT,
            updated_at TEXT NOT NULL
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS commodity_news_raw (
            id TEXT PRIMARY KEY,
            commodity TEXT NOT NULL,
            ticker TEXT NOT NULL,
            published_at TEXT,
            source TEXT,
            url TEXT,
            headline TEXT NOT NULL,
            sentiment REAL NOT NULL DEFAULT 0.0,
            event_tags TEXT,
            fetched_at TEXT NOT NULL
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS commodity_news_features (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            commodity TEXT NOT NULL,
            news_count_7d REAL NOT NULL DEFAULT 0.0,
            news_count_30d REAL NOT NULL DEFAULT 0.0,
            sent_mean_7d REAL NOT NULL DEFAULT 0.0,
            sent_mean_30d REAL NOT NULL DEFAULT 0.0,
            sent_shock REAL NOT NULL DEFAULT 0.0,
            event_score REAL NOT NULL DEFAULT 0.0,
            top_headlines_json TEXT,
            PRIMARY KEY (date, ticker)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS commodity_recommendations (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            commodity TEXT NOT NULL,
            action TEXT NOT NULL,
            score REAL NOT NULL,
            confidence REAL NOT NULL,
            target_weight REAL NOT NULL,
            invested_weight REAL NOT NULL,
            reasons TEXT,
            trading212_name TEXT,
            trading212_ticker TEXT,
            PRIMARY KEY (asof_date, ticker)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS commodity_model_holdings (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            commodity TEXT NOT NULL,
            target_weight REAL NOT NULL,
            invested_weight REAL NOT NULL,
            price REAL NOT NULL,
            shares REAL NOT NULL,
            value REAL NOT NULL,
            trading212_name TEXT,
            trading212_ticker TEXT,
            reasons TEXT,
            confidence REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (asof_date, ticker)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS commodity_model_trades (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            trade_action TEXT NOT NULL,
            shares_delta REAL NOT NULL,
            est_notional REAL NOT NULL,
            PRIMARY KEY (asof_date, ticker)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS news_implication_raw (
            id TEXT PRIMARY KEY,
            event TEXT NOT NULL,
            published_at TEXT,
            source TEXT,
            url TEXT,
            headline TEXT NOT NULL,
            affected_ticker TEXT NOT NULL,
            direction REAL NOT NULL,
            magnitude REAL NOT NULL,
            why TEXT,
            fetched_at TEXT NOT NULL
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS news_implication_features (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            implication_score_7d REAL NOT NULL DEFAULT 0.0,
            implication_score_30d REAL NOT NULL DEFAULT 0.0,
            implication_count_7d REAL NOT NULL DEFAULT 0.0,
            implication_count_30d REAL NOT NULL DEFAULT 0.0,
            implication_events_json TEXT,
            PRIMARY KEY (date, ticker)
        )
        """)

def get_last_date_for_ticker(conn: sqlite3.Connection, ticker: str) -> str | None:
    cur = conn.execute(
        "SELECT MAX(date) FROM prices_daily WHERE ticker = ?",
        (ticker,)
    )
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else None
