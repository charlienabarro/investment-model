# src/news_sentiment.py — Phase 1: Google News RSS + VADER sentiment
"""
Fetches recent news headlines for each ticker via Google News RSS,
scores sentiment locally with VADER, and stores results in the DB.

No API keys needed. No cost. Runs as part of the monthly pipeline.

Usage:
    from .news_sentiment import update_news_sentiment
    update_news_sentiment()  # call in run_pipeline after update_all_prices
"""
from __future__ import annotations

import re
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import quote

import pandas as pd

from .db import get_conn
from .universe import get_universe


# ── Ticker → search-friendly name mapping ──────────────────
# Google News works better with company names than ticker symbols

TICKER_NAME_MAP = {
    # Big tech
    "aapl.us": "Apple AAPL",
    "msft.us": "Microsoft MSFT",
    "googl.us": "Google Alphabet GOOGL",
    "amzn.us": "Amazon AMZN",
    "meta.us": "Meta Platforms META",
    "nvda.us": "Nvidia NVDA",
    "tsla.us": "Tesla TSLA",
    # ETFs — search by full name
    "spy.us": "S&P 500 SPY ETF",
    "qqq.us": "Nasdaq QQQ ETF",
    "iwm.us": "Russell 2000 IWM",
    "tlt.us": "Treasury bonds TLT",
    "gld.us": "Gold GLD ETF",
    "slv.us": "Silver SLV ETF",
    "hyg.us": "High yield bonds HYG",
}


def _search_name(ticker: str) -> str:
    """Convert ticker to a Google News search query."""
    if ticker in TICKER_NAME_MAP:
        return TICKER_NAME_MAP[ticker]
    # Strip .us suffix, uppercase
    clean = ticker.replace(".us", "").upper()
    return f"{clean} stock"


# ── DB schema ──────────────────────────────────────────────

def init_news_tables() -> None:
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS news_raw (
            id TEXT PRIMARY KEY,
            published_at TEXT,
            source TEXT,
            url TEXT,
            headline TEXT,
            ticker TEXT,
            sentiment REAL,
            fetched_at TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS news_daily_features (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            news_count_7d INTEGER DEFAULT 0,
            news_count_30d INTEGER DEFAULT 0,
            sent_mean_7d REAL DEFAULT 0.0,
            sent_mean_30d REAL DEFAULT 0.0,
            sent_shock REAL DEFAULT 0.0,
            PRIMARY KEY (date, ticker)
        )
        """)


# ── Google News RSS fetching ───────────────────────────────

def _fetch_google_news_rss(query: str, days_back: int = 30) -> List[Dict]:
    """
    Fetch headlines from Google News RSS for a search query.
    Returns list of {title, link, published, source}.
    """
    import feedparser

    encoded = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}+when:{days_back}d&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(url)
    except Exception:
        return []

    items = []
    for entry in feed.get("entries", []):
        title = entry.get("title", "")
        link = entry.get("link", "")
        source = entry.get("source", {}).get("title", "unknown")

        # Parse published date
        published = None
        if "published_parsed" in entry and entry["published_parsed"]:
            try:
                published = datetime(*entry["published_parsed"][:6]).isoformat()
            except Exception:
                pass

        if not published:
            published = datetime.utcnow().isoformat()

        if title:
            items.append({
                "title": title,
                "link": link,
                "published": published,
                "source": source,
            })

    return items


# ── VADER sentiment scoring ────────────────────────────────

def _score_sentiment(headline: str) -> float:
    """
    Score a headline using VADER. Returns compound score (-1 to +1).
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(headline)
    return float(scores["compound"])


# ── Main pipeline functions ────────────────────────────────

def _make_id(ticker: str, headline: str, url: str) -> str:
    """Deterministic ID to avoid duplicates."""
    raw = f"{ticker}|{headline}|{url}"
    return hashlib.md5(raw.encode()).hexdigest()


def fetch_and_store_news(tickers: List[str] = None, days_back: int = 30) -> int:
    """
    Fetch news for all tickers (or a subset), score sentiment, store in DB.
    Returns count of new articles stored.
    """
    init_news_tables()

    if tickers is None:
        tickers = get_universe()

    # Only fetch for a manageable subset — top candidates + ETFs
    # For 500+ tickers, we'd be rate-limited. Focus on ETFs + top ~50 stocks.
    priority_tickers = [t for t in tickers if t in TICKER_NAME_MAP]
    other_tickers = [t for t in tickers if t not in TICKER_NAME_MAP]

    # Take all mapped tickers + first 40 unmapped ones
    fetch_list = priority_tickers + other_tickers[:40]

    now = datetime.utcnow().isoformat()
    new_count = 0

    for ticker in fetch_list:
        query = _search_name(ticker)
        articles = _fetch_google_news_rss(query, days_back=days_back)

        rows = []
        for art in articles:
            art_id = _make_id(ticker, art["title"], art["link"])
            sentiment = _score_sentiment(art["title"])

            rows.append((
                art_id,
                art["published"],
                art["source"],
                art["link"],
                art["title"],
                ticker,
                sentiment,
                now,
            ))

        if rows:
            with get_conn() as conn:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO news_raw
                    (id, published_at, source, url, headline, ticker, sentiment, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            new_count += len(rows)

        # Be polite to Google — small delay between requests
        time.sleep(0.5)

    print(f"[OK] Fetched {new_count} news articles for {len(fetch_list)} tickers")
    return new_count


def build_news_features(asof_date: str = None) -> pd.DataFrame:
    """
    Aggregate raw news into per-ticker daily features.
    If asof_date is None, uses today.

    Returns DataFrame with columns:
        date, ticker, news_count_7d, news_count_30d,
        sent_mean_7d, sent_mean_30d, sent_shock
    """
    init_news_tables()

    if asof_date is None:
        asof_date = datetime.utcnow().date().isoformat()

    asof_dt = pd.to_datetime(asof_date)
    date_7d = (asof_dt - timedelta(days=7)).isoformat()
    date_30d = (asof_dt - timedelta(days=30)).isoformat()

    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT ticker, published_at, sentiment FROM news_raw WHERE published_at >= ?",
            conn,
            params=(date_30d,),
        )

    if df.empty:
        return pd.DataFrame(columns=[
            "date", "ticker", "news_count_7d", "news_count_30d",
            "sent_mean_7d", "sent_mean_30d", "sent_shock",
        ])

    df["published_at"] = pd.to_datetime(df["published_at"])

    # 30-day aggregates
    agg_30d = df.groupby("ticker").agg(
        news_count_30d=("sentiment", "count"),
        sent_mean_30d=("sentiment", "mean"),
    ).reset_index()

    # 7-day aggregates
    recent = df[df["published_at"] >= pd.to_datetime(date_7d)]
    if recent.empty:
        agg_7d = pd.DataFrame({"ticker": agg_30d["ticker"], "news_count_7d": 0, "sent_mean_7d": 0.0})
    else:
        agg_7d = recent.groupby("ticker").agg(
            news_count_7d=("sentiment", "count"),
            sent_mean_7d=("sentiment", "mean"),
        ).reset_index()

    # Merge
    out = agg_30d.merge(agg_7d, on="ticker", how="left").fillna(0.0)

    # Sentiment shock: recent sentiment vs longer-term
    out["sent_shock"] = out["sent_mean_7d"] - out["sent_mean_30d"]

    out["date"] = asof_date

    # Store in DB
    rows = out.to_records(index=False)
    with get_conn() as conn:
        for r in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO news_daily_features
                (date, ticker, news_count_7d, news_count_30d, sent_mean_7d, sent_mean_30d, sent_shock)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (r.date, r.ticker, int(r.news_count_7d), int(r.news_count_30d),
                 float(r.sent_mean_7d), float(r.sent_mean_30d), float(r.sent_shock)),
            )

    return out[["date", "ticker", "news_count_7d", "news_count_30d",
                "sent_mean_7d", "sent_mean_30d", "sent_shock"]]


def update_news_sentiment() -> pd.DataFrame:
    """
    Full pipeline step: fetch news, build features, return feature DataFrame.
    Call this in run_pipeline().
    """
    fetch_and_store_news(days_back=30)
    features = build_news_features()
    return features