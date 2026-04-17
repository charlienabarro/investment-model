from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from .db import get_conn
from .universe import get_universe

CASH_TICKER = "cash"

EVENT_QUERIES = [
    "jet fuel shortage Europe aviation fuel kerosene",
    "heating oil futures diesel jet fuel inventories",
    "OPEC supply cut crude oil inventory sanctions",
    "Brent crude Strait of Hormuz Europe fuel imports",
    "natural gas storage inventory LNG weather Europe",
    "copper inventory China demand stimulus",
    "wheat exports Ukraine drought grain prices",
    "corn drought ethanol harvest futures",
    "soybean China demand drought harvest",
    "cocoa shortage West Africa harvest",
    "coffee crop Brazil drought robusta arabica",
    "sugar crop India Brazil export futures",
    "cotton crop drought demand futures",
    "gold real yields inflation Fed safe haven",
    "silver industrial demand solar supply",
]

IMPLICATION_RULES = [
    {
        "event": "jet_fuel_shortage",
        "patterns": ["jet fuel", "aviation fuel", "kerosene", "fuel shortage", "six weeks", "heating oil"],
        "positive": ["ho=f", "cl=f", "bz=f", "uso.us", "xom.us", "cvx.us", "cop.us", "psx.us", "slb.us", "oxy.us"],
        "negative": ["ba.us", "fdx.us", "ups.us", "mar.us", "bkng.us", "uber.us"],
        "why": "Fuel scarcity can lift refined fuel/heating-oil prices and energy/refiner margins, while pressuring transport/travel costs.",
    },
    {
        "event": "oil_supply_tightness",
        "patterns": ["opec", "supply cut", "output cut", "sanctions", "inventory draw", "crude inventory", "hormuz"],
        "positive": ["cl=f", "bz=f", "ho=f", "uso.us", "xom.us", "cvx.us", "cop.us", "oxy.us", "slb.us", "psx.us"],
        "negative": ["fdx.us", "ups.us", "ba.us", "mar.us", "bkng.us", "mcd.us", "sbux.us"],
        "why": "Tighter oil supply usually supports crude/refined-product prices and energy producers, but raises input costs for transport and consumer sectors.",
    },
    {
        "event": "gas_storage_tightness",
        "patterns": ["natural gas", "gas storage", "lng", "gas inventory", "cold weather", "heatwave"],
        "positive": ["ng=f", "ung.us", "xom.us", "cvx.us", "cop.us", "kmi.us", "bkr.us"],
        "negative": ["d.us", "so.us", "duk.us", "nee.us", "aep.us"],
        "why": "Gas shortages or stronger LNG/weather demand can lift gas prices and gas producers, while pressuring fuel-consuming utilities.",
    },
    {
        "event": "copper_demand_or_shortage",
        "patterns": ["copper", "china demand", "china stimulus", "inventory", "supply disruption", "mine strike"],
        "positive": ["hg=f", "cper.us", "copx.us", "cat.us", "de.us", "slb.us"],
        "negative": [],
        "why": "Copper demand or supply tightness supports copper and miners, and can signal stronger industrial/infrastructure activity.",
    },
    {
        "event": "grain_crop_stress",
        "patterns": ["wheat", "corn", "soybean", "grain", "drought", "crop", "harvest", "export ban"],
        "positive": ["zw=f", "zc=f", "zs=f", "dba.us"],
        "negative": ["mdlz.us", "pep.us", "ko.us", "mcd.us", "sbux.us", "cost.us", "wmt.us"],
        "why": "Crop stress can lift grain/agriculture prices but squeeze food, restaurant, and retailer margins.",
    },
    {
        "event": "softs_shortage",
        "patterns": ["cocoa", "coffee", "sugar", "cotton", "arabica", "robusta", "crop disease"],
        "positive": ["cc=f", "kc=f", "sb=f", "ct=f", "dba.us"],
        "negative": ["mdlz.us", "pep.us", "ko.us", "sbux.us", "mcd.us"],
        "why": "Soft-commodity shortages can lift futures/ETCs, while increasing input costs for food and beverage companies.",
    },
    {
        "event": "precious_metals_support",
        "patterns": ["gold", "silver", "inflation", "fed", "rate cut", "real yields", "safe haven", "central bank gold"],
        "positive": ["gc=f", "si=f", "gld.us", "iau.us", "slv.us"],
        "negative": [],
        "why": "Inflation, rate-cut expectations, safe-haven demand, or central-bank buying can support precious metals.",
    },
]


def init_implication_tables() -> None:
    with get_conn() as conn:
        conn.execute(
            """
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
            """
        )
        conn.execute(
            """
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
            """
        )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _published_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date().isoformat()
    except Exception:
        return None


def _fetch_google_news_rss(query: str, max_items: int = 12, timeout: float = 8.0) -> list[dict]:
    url = "https://news.google.com/rss/search?q=" + quote_plus(query) + "&hl=en-GB&gl=GB&ceid=GB:en"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    root = ET.fromstring(payload)
    out = []
    for item in root.findall("./channel/item")[:max_items]:
        title = _normalise(item.findtext("title") or "")
        if not title:
            continue
        out.append(
            {
                "headline": title,
                "url": _normalise(item.findtext("link") or ""),
                "source": _normalise(item.findtext("source") or "Google News"),
                "published_at": _published_date(item.findtext("pubDate")) or datetime.now(timezone.utc).date().isoformat(),
            }
        )
    return out


def _article_id(event: str, ticker: str, url: str, headline: str, direction: float) -> str:
    raw = f"{event}|{ticker}|{direction}|{url}|{headline}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def _match_rules(headline: str, available: set[str]) -> list[tuple[str, str, float, str]]:
    text = f" {headline.lower()} "
    rows = []
    for rule in IMPLICATION_RULES:
        if not any(p in text for p in rule["patterns"]):
            continue
        if rule["event"] == "grain_crop_stress" and not any(p in text for p in ["wheat", "corn", "soybean", "grain", "crop", "harvest"]):
            continue
        if rule["event"] == "softs_shortage" and not any(p in text for p in ["cocoa", "coffee", "sugar", "cotton", "arabica", "robusta"]):
            continue
        matched_strength = sum(1 for p in rule["patterns"] if p in text)
        magnitude = min(1.0, 0.45 + 0.15 * matched_strength)
        for ticker in rule["positive"]:
            if ticker in available:
                rows.append((rule["event"], ticker, magnitude, rule["why"]))
        for ticker in rule["negative"]:
            if ticker in available:
                rows.append((rule["event"], ticker, -magnitude, rule["why"]))
    return rows


def fetch_and_store_implication_news(extra_tickers: list[str] | None = None, days_back: int = 30, per_query: int = 12) -> int:
    init_implication_tables()
    universe = {str(t).lower() for t in get_universe()}
    if extra_tickers:
        universe |= {str(t).lower() for t in extra_tickers}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []

    for query in EVENT_QUERIES:
        try:
            articles = _fetch_google_news_rss(query, max_items=per_query)
        except Exception as e:
            print(f"[WARN] Implication news failed for '{query}': {type(e).__name__}: {e}")
            continue
        for article in articles:
            if article["published_at"] < cutoff:
                continue
            for event, ticker, direction, why in _match_rules(article["headline"], universe):
                rows.append(
                    (
                        _article_id(event, ticker, article["url"], article["headline"], direction),
                        event,
                        article["published_at"],
                        article["source"],
                        article["url"],
                        article["headline"],
                        ticker,
                        1.0 if direction > 0 else -1.0,
                        abs(float(direction)),
                        why,
                        fetched_at,
                    )
                )
        time.sleep(0.15)

    if not rows:
        print("[WARN] Implication news: no directional event rows found")
        return 0

    with get_conn() as conn:
        conn.execute("DELETE FROM news_implication_raw")
        conn.executemany(
            """
            INSERT INTO news_implication_raw
              (id, event, published_at, source, url, headline, affected_ticker, direction, magnitude, why, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              event=excluded.event,
              published_at=excluded.published_at,
              source=excluded.source,
              url=excluded.url,
              headline=excluded.headline,
              affected_ticker=excluded.affected_ticker,
              direction=excluded.direction,
              magnitude=excluded.magnitude,
              why=excluded.why,
              fetched_at=excluded.fetched_at
            """,
            rows,
        )
    print(f"[OK] Implication news: stored {len(rows)} directional rows")
    return len(rows)


def build_implication_features(asof_date: str | None = None) -> pd.DataFrame:
    init_implication_tables()
    if asof_date is None:
        asof_date = datetime.now(timezone.utc).date().isoformat()
    asof = pd.to_datetime(asof_date)
    start_30 = (asof - pd.Timedelta(days=30)).date().isoformat()
    with get_conn() as conn:
        raw = pd.read_sql_query(
            """
            SELECT event, published_at, headline, affected_ticker, direction, magnitude, why
            FROM news_implication_raw
            WHERE published_at IS NOT NULL AND published_at >= ? AND published_at <= ?
            """,
            conn,
            params=(start_30, asof_date),
        )

    columns = [
        "date", "ticker", "implication_score_7d", "implication_score_30d",
        "implication_count_7d", "implication_count_30d", "implication_events_json",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    raw["published_at"] = pd.to_datetime(raw["published_at"], errors="coerce")
    raw = raw.dropna(subset=["published_at"])
    raw["score"] = raw["direction"].astype(float) * raw["magnitude"].astype(float)
    recent = raw[raw["published_at"] >= asof - pd.Timedelta(days=7)].copy()

    rows = []
    for ticker, g30 in raw.groupby("affected_ticker"):
        g7 = recent[recent["affected_ticker"] == ticker]
        events = (
            g30.sort_values("published_at", ascending=False)
            .head(6)[["event", "headline", "score", "why"]]
            .to_dict(orient="records")
        )
        rows.append(
            {
                "date": asof_date,
                "ticker": str(ticker).lower(),
                "implication_score_7d": float(g7["score"].sum()) if not g7.empty else 0.0,
                "implication_score_30d": float(g30["score"].sum()),
                "implication_count_7d": float(len(g7)),
                "implication_count_30d": float(len(g30)),
                "implication_events_json": json.dumps(events),
            }
        )

    out = pd.DataFrame(rows, columns=columns)
    with get_conn() as conn:
        conn.execute("DELETE FROM news_implication_features WHERE date = ?", (asof_date,))
        conn.executemany(
            """
            INSERT INTO news_implication_features
              (date, ticker, implication_score_7d, implication_score_30d, implication_count_7d, implication_count_30d, implication_events_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.date,
                    r.ticker,
                    float(r.implication_score_7d),
                    float(r.implication_score_30d),
                    float(r.implication_count_7d),
                    float(r.implication_count_30d),
                    r.implication_events_json,
                )
                for r in out.itertuples(index=False)
            ],
        )
    return out


def update_implication_news(asof_date: str | None = None, extra_tickers: list[str] | None = None) -> pd.DataFrame:
    fetch_and_store_implication_news(extra_tickers=extra_tickers)
    return build_implication_features(asof_date=asof_date)
