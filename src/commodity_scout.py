from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .backtest import run_recommendation_backtest
from .config import MIN_TRADE_GBP, PORTFOLIO_VALUE
from .db import get_conn, get_last_date_for_ticker, init_db
from .features import build_feature_frame, add_cross_sectional_zscores, add_sector_relative_features
from .pipeline import _iso_date, _resolve_portfolio_value, upsert_prices
from .stooq_data import fetch_daily_with_fallback
from .implication_news import update_implication_news, build_implication_features

CASH_TICKER = "cash"
MIN_COMMODITY_WEIGHT = 0.10
MAX_COMMODITY_WEIGHT = 0.35
MIN_POSITIONS = 3
MAX_POSITIONS = 5


@dataclass(frozen=True)
class CommodityInstrument:
    ticker: str
    commodity: str
    display_name: str
    trading212_name: str
    trading212_ticker: str
    search_terms: tuple[str, ...]


COMMODITY_INSTRUMENTS: tuple[CommodityInstrument, ...] = (
    # Precious metals: ETF/ETC investable proxies plus underlying futures signals.
    CommodityInstrument("gld.us", "gold", "Gold", "SPDR Gold Shares", "GLD", ("gold price", "gold ETF", "central bank gold", "Fed inflation gold")),
    CommodityInstrument("iau.us", "gold", "Gold", "iShares Gold Trust", "IAU", ("gold price", "gold ETF", "central bank gold", "Fed inflation gold")),
    CommodityInstrument("gc=f", "gold", "Gold Futures", "iShares Gold Trust or WisdomTree Physical Gold", "IAU / PHGP", ("gold futures", "gold price", "central bank gold", "real yields gold")),
    CommodityInstrument("slv.us", "silver", "Silver", "iShares Physical Silver", "SSLN", ("silver price", "silver demand", "silver supply", "industrial silver")),
    CommodityInstrument("si=f", "silver", "Silver Futures", "iShares Physical Silver", "SSLN", ("silver futures", "silver price", "industrial silver demand", "silver supply")),
    CommodityInstrument("pl=f", "platinum", "Platinum Futures", "WisdomTree Physical Platinum", "PHPT", ("platinum price", "platinum supply", "platinum auto demand", "platinum futures")),
    CommodityInstrument("pa=f", "palladium", "Palladium Futures", "WisdomTree Physical Palladium", "PHPD", ("palladium price", "palladium supply", "palladium auto demand", "palladium futures")),

    # Broad baskets.
    CommodityInstrument("dbc.us", "broad_commodities", "Broad Commodities", "Invesco DB Commodity Index Tracking Fund", "DBC", ("commodity prices", "commodities index", "inflation commodities", "China commodity demand")),
    CommodityInstrument("pdbc.us", "broad_commodities", "Broad Commodities", "Invesco Optimum Yield Diversified Commodity Strategy", "PDBC", ("commodity prices", "commodities index", "inflation commodities", "China commodity demand")),
    CommodityInstrument("djp.us", "broad_commodities", "Broad Commodities", "iPath Bloomberg Commodity Index Total Return", "DJP", ("commodity prices", "commodities index", "inflation commodities", "China commodity demand")),

    # Energy, now split into the specific products that move differently.
    CommodityInstrument("uso.us", "wti_crude", "WTI Crude Oil", "United States Oil Fund or WisdomTree WTI Crude Oil", "USO / CRUD", ("WTI crude oil", "OPEC supply cut", "crude inventory", "sanctions oil")),
    CommodityInstrument("cl=f", "wti_crude", "WTI Crude Oil Futures", "WisdomTree WTI Crude Oil", "CRUD", ("WTI crude futures", "OPEC supply cut", "crude inventory", "sanctions oil")),
    CommodityInstrument("bz=f", "brent_crude", "Brent Crude Oil Futures", "WisdomTree Brent Crude Oil", "BRNT", ("Brent crude futures", "Brent oil supply", "Europe oil imports", "Strait of Hormuz oil")),
    CommodityInstrument("ho=f", "heating_oil_jet_fuel_proxy", "Heating Oil / Jet Fuel Proxy", "WisdomTree Heating Oil", "HEAT", ("jet fuel shortage Europe", "heating oil futures", "aviation fuel shortage", "diesel jet fuel inventories")),
    CommodityInstrument("rb=f", "gasoline", "Gasoline Futures", "WisdomTree Gasoline", "UGAS", ("gasoline futures", "gasoline inventory", "refinery outage gasoline", "US gasoline demand")),
    CommodityInstrument("ung.us", "natural_gas", "Natural Gas", "United States Natural Gas Fund", "UNG", ("natural gas price", "gas storage inventory", "LNG demand", "weather natural gas")),
    CommodityInstrument("ng=f", "natural_gas", "Natural Gas Futures", "WisdomTree Natural Gas", "NGAS", ("natural gas futures", "gas storage inventory", "LNG demand", "weather natural gas")),

    # Industrial metals and miners.
    CommodityInstrument("cper.us", "copper", "Copper", "United States Copper Index Fund", "CPER", ("copper price", "China copper demand", "copper inventory", "copper supply")),
    CommodityInstrument("hg=f", "copper", "Copper Futures", "WisdomTree Copper", "COPA", ("copper futures", "China copper demand", "copper inventory", "copper supply")),
    CommodityInstrument("copx.us", "copper_miners", "Copper Miners", "Global X Copper Miners ETF", "COPX", ("copper miners", "copper price", "China copper demand", "copper supply")),

    # Agriculture and soft commodities.
    CommodityInstrument("dba.us", "agriculture_basket", "Agriculture Basket", "Invesco DB Agriculture Fund", "DBA", ("agriculture commodities", "drought crops", "grain prices", "food inflation")),
    CommodityInstrument("zc=f", "corn", "Corn Futures", "WisdomTree Corn", "CORN", ("corn futures", "corn harvest", "corn drought", "ethanol corn demand")),
    CommodityInstrument("zw=f", "wheat", "Wheat Futures", "WisdomTree Wheat", "WEAT", ("wheat futures", "wheat exports", "Ukraine wheat", "wheat drought")),
    CommodityInstrument("zs=f", "soybeans", "Soybean Futures", "WisdomTree Soybeans", "SOYB", ("soybean futures", "soybean demand China", "soybean harvest", "soybean drought")),
    CommodityInstrument("kc=f", "coffee", "Coffee Futures", "WisdomTree Coffee", "COFF", ("coffee futures", "coffee crop", "Brazil coffee drought", "robusta arabica coffee")),
    CommodityInstrument("cc=f", "cocoa", "Cocoa Futures", "WisdomTree Cocoa", "COCO", ("cocoa futures", "cocoa shortage", "West Africa cocoa", "cocoa harvest")),
    CommodityInstrument("sb=f", "sugar", "Sugar Futures", "WisdomTree Sugar", "SUGA", ("sugar futures", "sugar crop", "Brazil sugar", "India sugar export")),
    CommodityInstrument("ct=f", "cotton", "Cotton Futures", "WisdomTree Cotton", "COTN", ("cotton futures", "cotton crop", "cotton demand", "cotton drought")),
    CommodityInstrument("le=f", "live_cattle", "Live Cattle Futures", "Live cattle ETC / closest agriculture ETC", "CATTLE", ("live cattle futures", "cattle herd", "beef supply", "cattle prices")),
    CommodityInstrument("he=f", "lean_hogs", "Lean Hogs Futures", "Lean hogs ETC / closest agriculture ETC", "HOGS", ("lean hog futures", "pork supply", "hog prices", "China pork demand")),
    CommodityInstrument("oj=f", "orange_juice", "Orange Juice Futures", "Orange juice ETC / closest agriculture ETC", "OJ", ("orange juice futures", "orange crop disease", "Florida oranges", "orange juice shortage")),
)

COMMODITY_UNIVERSE = [x.ticker for x in COMMODITY_INSTRUMENTS]
MACRO_ANCHORS = ["spy.us", "tlt.us", "shy.us", "lqd.us", "hyg.us"]
INSTRUMENT_BY_TICKER = {x.ticker: x for x in COMMODITY_INSTRUMENTS}
NON_SELECTABLE_SIGNAL_TICKERS = {"le=f", "he=f", "oj=f"}
SPOT_FUTURES_PAIRS = {
    "wti_crude": ("uso.us", "cl=f"),
    "gold": ("iau.us", "gc=f"),
    "silver": ("slv.us", "si=f"),
    "natural_gas": ("ung.us", "ng=f"),
    "copper": ("cper.us", "hg=f"),
}

EVENT_KEYWORDS = {
    "opec": ["opec"],
    "supply_cut": ["supply cut", "production cut", "output cut", "supply disruption", "supply crunch"],
    "sanctions": ["sanction", "embargo", "export ban"],
    "drought": ["drought", "heatwave", "crop failure"],
    "inventory": ["inventory", "stockpile", "storage"],
    "fuel_shortage": ["jet fuel", "aviation fuel", "kerosene", "fuel shortage", "weeks of fuel"],
    "refinery": ["refinery", "refineries", "refining margin", "crack spread"],
    "china_demand": ["china demand", "chinese demand", "china stimulus", "china property"],
    "fed": ["federal reserve", " fed ", "rate cut", "rate hike", "interest rate"],
    "inflation": ["inflation", "cpi", "ppi", "prices rose"],
}

COMMODITY_EVENT_IMPACTS = {
    ("wti_crude", "opec"): 0.20,
    ("wti_crude", "supply_cut"): 0.18,
    ("wti_crude", "sanctions"): 0.15,
    ("wti_crude", "inventory"): 0.10,
    ("brent_crude", "opec"): 0.20,
    ("brent_crude", "supply_cut"): 0.18,
    ("brent_crude", "sanctions"): 0.15,
    ("brent_crude", "inventory"): 0.10,
    ("heating_oil_jet_fuel_proxy", "fuel_shortage"): 0.22,
    ("heating_oil_jet_fuel_proxy", "refinery"): 0.16,
    ("heating_oil_jet_fuel_proxy", "inventory"): 0.14,
    ("gasoline", "refinery"): 0.18,
    ("gasoline", "inventory"): 0.16,
    ("natural_gas", "inventory"): 0.18,
    ("natural_gas", "drought"): 0.05,
    ("gold", "fed"): 0.18,
    ("gold", "inflation"): 0.15,
    ("silver", "fed"): 0.12,
    ("silver", "inflation"): 0.12,
    ("copper", "china_demand"): 0.20,
    ("copper", "inventory"): 0.12,
    ("copper_miners", "china_demand"): 0.18,
    ("wheat", "drought"): 0.20,
    ("wheat", "sanctions"): 0.15,
    ("corn", "drought"): 0.20,
    ("soybeans", "drought"): 0.18,
    ("coffee", "drought"): 0.20,
    ("cocoa", "drought"): 0.16,
    ("agriculture_basket", "drought"): 0.18,
}

POSITIVE_WORDS = {
    "rally", "rises", "rise", "gains", "gain", "surge", "surges", "strong", "deficit", "shortage",
    "cut", "cuts", "demand", "stimulus", "inflation", "safe-haven", "safe haven", "tight", "bullish",
}
NEGATIVE_WORDS = {
    "falls", "fall", "drops", "drop", "slump", "weak", "surplus", "glut", "recession", "bearish",
    "inventory build", "higher supply", "slowdown", "rate hike", "dollar strength", "risk-off",
}

_FINBERT = None
_FINBERT_AVAILABLE: bool | None = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _instrument(ticker: str) -> CommodityInstrument:
    return INSTRUMENT_BY_TICKER[str(ticker).lower()]


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _article_id(url: str, headline: str) -> str:
    raw = f"{url}|{headline}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def _sentiment(text: str) -> float:
    global _FINBERT, _FINBERT_AVAILABLE

    # TODO: FinBERT requires ~500MB model download on first use. Consider
    # caching the model locally in data/finbert/. Set HF_HOME env var to
    # control cache location.
    if _FINBERT_AVAILABLE is not False:
        try:
            if _FINBERT is None:
                from transformers import pipeline
                _FINBERT = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                _FINBERT_AVAILABLE = True
            result = _FINBERT(text or "")[0]
            label = str(result.get("label", "")).lower()
            score = float(result.get("score", 0.0))
            if "positive" in label:
                return float(np.clip(2.0 * score - 1.0, -1.0, 1.0))
            if "negative" in label:
                return float(np.clip(1.0 - 2.0 * score, -1.0, 1.0))
            if "neutral" in label:
                return 0.0
        except Exception:
            _FINBERT_AVAILABLE = False

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return float(SentimentIntensityAnalyzer().polarity_scores(text or "")["compound"])
    except Exception:
        s = f" {(text or '').lower()} "
        pos = sum(1 for word in POSITIVE_WORDS if word in s)
        neg = sum(1 for word in NEGATIVE_WORDS if word in s)
        if pos == 0 and neg == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos - neg) / max(pos + neg, 1)))


def _event_tags(text: str) -> list[str]:
    s = f" {(text or '').lower()} "
    tags = []
    for tag, needles in EVENT_KEYWORDS.items():
        if any(n in s for n in needles):
            tags.append(tag)
    return tags


def _parse_rss_datetime(value: str | None) -> str | None:
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
        xml_bytes = resp.read()
    root = ET.fromstring(xml_bytes)
    out = []
    for item in root.findall("./channel/item")[:max_items]:
        title = _normalise_text(item.findtext("title") or "")
        link = _normalise_text(item.findtext("link") or "")
        pub_date = _parse_rss_datetime(item.findtext("pubDate"))
        source = _normalise_text(item.findtext("source") or "Google News")
        if title:
            out.append({"headline": title, "url": link, "published_at": pub_date, "source": source})
    return out


def update_commodity_prices(include_macro: bool = False) -> None:
    tickers = list(COMMODITY_UNIVERSE)
    if include_macro:
        tickers.extend([x for x in MACRO_ANCHORS if x not in tickers])

    inserted = 0
    for ticker in tickers:
        df, source, err = fetch_daily_with_fallback(ticker)
        if df.empty:
            reason = f" ({err})" if err else ""
            print(f"[WARN] Commodity price: no data for {ticker}{reason}")
            continue
        with get_conn() as conn:
            last_date = get_last_date_for_ticker(conn, ticker)
        if last_date is not None:
            df = df[pd.to_datetime(df["date"]) > pd.to_datetime(last_date)]
        n = upsert_prices(ticker, df)
        inserted += n
        with get_conn() as conn:
            up_to = get_last_date_for_ticker(conn, ticker)
        msg = f"inserted or updated {n} rows" if n else "up to date"
        print(f"[OK] Commodity price {ticker}: {msg} via {source} (up to {up_to})")
    print(f"[OK] Commodity price refresh complete ({inserted} new rows)")


def fetch_and_store_commodity_news(days_back: int = 30, per_query: int = 10) -> int:
    fetched_at = _now_utc().isoformat()
    cutoff = (_now_utc() - timedelta(days=days_back)).date().isoformat()
    rows = []
    seen = set()

    for inst in COMMODITY_INSTRUMENTS:
        for query in inst.search_terms:
            try:
                articles = _fetch_google_news_rss(query, max_items=per_query)
            except Exception as e:
                print(f"[WARN] Commodity news failed for '{query}': {type(e).__name__}: {e}")
                continue
            for a in articles:
                pub = a.get("published_at") or _now_utc().date().isoformat()
                if pub < cutoff:
                    continue
                headline = _normalise_text(a.get("headline", ""))
                url = _normalise_text(a.get("url", ""))
                key = _article_id(url, headline)
                dedupe_key = (inst.commodity, key)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                sent = _sentiment(headline)
                tags = _event_tags(headline)
                rows.append((key, inst.commodity, inst.ticker, pub, a.get("source") or "Google News", url, headline, sent, json.dumps(tags), fetched_at))

    if not rows:
        print("[WARN] Commodity news: no fresh articles fetched; continuing with existing news/fallback features")
        return 0

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO commodity_news_raw
              (id, commodity, ticker, published_at, source, url, headline, sentiment, event_tags, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              commodity=excluded.commodity,
              ticker=excluded.ticker,
              published_at=excluded.published_at,
              source=excluded.source,
              url=excluded.url,
              headline=excluded.headline,
              sentiment=excluded.sentiment,
              event_tags=excluded.event_tags,
              fetched_at=excluded.fetched_at
            """,
            rows,
        )
    print(f"[OK] Commodity news: stored {len(rows)} rows")
    return len(rows)


def load_prices_for_commodities(include_macro: bool = True) -> pd.DataFrame:
    tickers = list(COMMODITY_UNIVERSE)
    if include_macro:
        tickers.extend([x for x in MACRO_ANCHORS if x not in tickers])
    placeholders = ",".join(["?"] * len(tickers))
    with get_conn() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT date, ticker, open, high, low, close, volume
            FROM prices_daily
            WHERE ticker IN ({placeholders})
            ORDER BY date ASC
            """,
            conn,
            params=tickers,
        )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def _latest_price_map() -> dict[str, float]:
    placeholders = ",".join(["?"] * len(COMMODITY_UNIVERSE))
    with get_conn() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT p1.ticker, p1.close
            FROM prices_daily p1
            JOIN (
                SELECT ticker, MAX(date) AS max_date
                FROM prices_daily
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
            ) p2 ON p1.ticker = p2.ticker AND p1.date = p2.max_date
            """,
            conn,
            params=COMMODITY_UNIVERSE,
        )
    out = {str(r.ticker).lower(): float(r.close) for r in df.itertuples()}
    out[CASH_TICKER] = 1.0
    return out


def build_commodity_news_features(asof_date: str | None = None) -> pd.DataFrame:
    with get_conn() as conn:
        if asof_date is None:
            row = conn.execute("SELECT MAX(date) FROM prices_daily WHERE ticker IN (%s)" % ",".join(["?"] * len(COMMODITY_UNIVERSE)), COMMODITY_UNIVERSE).fetchone()
            asof_date = row[0] if row and row[0] else _now_utc().date().isoformat()
        raw = pd.read_sql_query(
            """
            SELECT commodity, ticker, published_at, source, url, headline, sentiment, event_tags
            FROM commodity_news_raw
            WHERE published_at IS NOT NULL AND published_at <= ?
            """,
            conn,
            params=(asof_date,),
        )

    asof_ts = pd.to_datetime(asof_date)
    feature_rows = []
    for inst in COMMODITY_INSTRUMENTS:
        g = raw[raw["commodity"] == inst.commodity].copy() if not raw.empty else pd.DataFrame()
        if not g.empty:
            g["published_at"] = pd.to_datetime(g["published_at"], errors="coerce")
            g = g.dropna(subset=["published_at"])
            g7 = g[g["published_at"] >= asof_ts - pd.Timedelta(days=7)]
            g30 = g[g["published_at"] >= asof_ts - pd.Timedelta(days=30)]
            sent7 = float(g7["sentiment"].mean()) if not g7.empty else 0.0
            sent30 = float(g30["sentiment"].mean()) if not g30.empty else 0.0
            event_score = 0.0
            for tags_json in g7["event_tags"].fillna("[]"):
                try:
                    tags = json.loads(tags_json)
                    for tag in tags:
                        event_score += COMMODITY_EVENT_IMPACTS.get((inst.commodity, str(tag)), 0.04)
                except Exception:
                    pass
            top = g30.sort_values("published_at", ascending=False).head(5)[["headline", "source", "url", "sentiment"]].to_dict(orient="records")
            row = {
                "date": asof_date,
                "ticker": inst.ticker,
                "commodity": inst.commodity,
                "news_count_7d": float(len(g7)),
                "news_count_30d": float(len(g30)),
                "sent_mean_7d": sent7,
                "sent_mean_30d": sent30,
                "sent_shock": sent7 - sent30,
                "event_score": float(min(event_score, 1.0)),
                "top_headlines_json": json.dumps(top),
            }
        else:
            row = {
                "date": asof_date,
                "ticker": inst.ticker,
                "commodity": inst.commodity,
                "news_count_7d": 0.0,
                "news_count_30d": 0.0,
                "sent_mean_7d": 0.0,
                "sent_mean_30d": 0.0,
                "sent_shock": 0.0,
                "event_score": 0.0,
                "top_headlines_json": "[]",
            }
        feature_rows.append(row)

    feats = pd.DataFrame(feature_rows)
    with get_conn() as conn:
        conn.execute("DELETE FROM commodity_news_features WHERE date = ?", (asof_date,))
        conn.executemany(
            """
            INSERT INTO commodity_news_features
              (date, ticker, commodity, news_count_7d, news_count_30d, sent_mean_7d, sent_mean_30d, sent_shock, event_score, top_headlines_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.date,
                    r.ticker,
                    r.commodity,
                    float(r.news_count_7d),
                    float(r.news_count_30d),
                    float(r.sent_mean_7d),
                    float(r.sent_mean_30d),
                    float(r.sent_shock),
                    float(r.event_score),
                    r.top_headlines_json,
                )
                for r in feats.itertuples(index=False)
            ],
        )
    return feats


def _safe_float(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(name, default))
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return (s - s.mean()) / (s.std() + 1e-12)


def _bounded_numeric(series: pd.Series, low: float, high: float, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).clip(lower=low, upper=high)


def _column_or_default(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index, dtype=float)


def add_commodity_roll_yield_features(feats: pd.DataFrame) -> pd.DataFrame:
    out = feats.copy()
    out["roll_yield"] = 0.0
    if out.empty or "close" not in out.columns:
        return out

    tmp = out[["date", "ticker", "close"]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["ticker"] = tmp["ticker"].astype(str).str.lower()
    px = tmp.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()

    for commodity, (spot_ticker, futures_ticker) in SPOT_FUTURES_PAIRS.items():
        if spot_ticker not in px.columns or futures_ticker not in px.columns:
            # TODO: replace ETF-vs-continuous-future proxy with true contract
            # curve data when a reliable free source is available.
            continue
        spot = pd.to_numeric(px[spot_ticker], errors="coerce")
        fut = pd.to_numeric(px[futures_ticker], errors="coerce")
        roll = ((spot - fut) / (spot + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mask = out["ticker"].astype(str).str.lower().map(
            lambda t: t in INSTRUMENT_BY_TICKER and _instrument(t).commodity == commodity
        )
        out.loc[mask, "roll_yield"] = out.loc[mask, "date"].map(roll).fillna(0.0).astype(float)

    return out


def _lightgbm_scores(feats: pd.DataFrame) -> pd.DataFrame | None:
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return None

    df = feats[feats["ticker"].isin(COMMODITY_UNIVERSE)].copy()
    if df.empty:
        return None
    df = df.sort_values(["ticker", "date"])
    df["fwd_21"] = df.groupby("ticker")["close"].shift(-21) / df["close"] - 1.0
    feature_cols = [
        c for c in [
            "mom_6_1", "mom_3_1", "mom_1_0", "ma_200_ratio", "trend_50_200", "vol_21", "vol_63",
            "maxdd_63", "sharpe_63", "rsi_14", "spy_mom_1m", "spy_vol_21", "gold_trend",
            "news_count_7d", "sent_mean_7d", "sent_shock", "event_score",
        ] if c in df.columns
    ]
    latest_date = df["date"].max()
    # 42-day embargo = 21-day forward-return window plus 21-day safety buffer.
    embargo = latest_date - pd.Timedelta(days=42)
    train = df[df["date"] < embargo].dropna(subset=["fwd_21"]).copy()
    train = train[train[feature_cols].notna().sum(axis=1) >= max(3, len(feature_cols) // 2)]
    if len(train) < 180 or len(feature_cols) < 5:
        return None
    X = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train["fwd_21"].astype(float)
    model = LGBMRegressor(n_estimators=80, learning_rate=0.04, max_depth=3, random_state=42, verbose=-1)
    try:
        model.fit(X, y)
        latest = df[df["date"] == latest_date].copy()
        latest["ml_score"] = model.predict(latest[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        return latest[["ticker", "ml_score"]]
    except Exception:
        return None


def _build_reasons(row: pd.Series, weight: float, cash_weight: float, score: float, confidence: float) -> str:
    inst = _instrument(str(row["ticker"]).lower())
    mom1 = _safe_float(row, "mom_1_0")
    mom3 = _safe_float(row, "mom_3_1")
    trend = _safe_float(row, "ma_200_ratio", 1.0)
    news7 = _safe_float(row, "sent_mean_7d")
    count7 = _safe_float(row, "news_count_7d")
    shock = _safe_float(row, "sent_shock")
    implication = _safe_float(row, "implication_score_7d")
    spy = _safe_float(row, "spy_mom_1m")
    vol = _safe_float(row, "vol_63") * math.sqrt(252)

    if mom1 > 0.03 or mom3 > 0.08 or trend > 1.03:
        price_reason = f"Price/trend: positive trend ({mom1*100:.1f}% over 1 month, {mom3*100:.1f}% over 3 months)."
    elif mom1 < -0.03 or trend < 0.97:
        price_reason = f"Price/trend: weak recent trend ({mom1*100:.1f}% over 1 month), so sizing is controlled."
    else:
        price_reason = f"Price/trend: steady rather than explosive ({mom1*100:.1f}% over 1 month)."

    if count7 > 0:
        tone = "positive" if news7 > 0.08 else ("negative" if news7 < -0.08 else "mixed")
        news_reason = f"News: {int(count7)} relevant recent stories with {tone} tone; sentiment shock {shock:+.2f}."
    else:
        news_reason = "News: no strong recent RSS signal found, so the model leaned more on price and macro data."
    if implication > 0.1:
        news_reason += f" Event implications are positive ({implication:+.1f}), meaning recent headlines point to tighter supply or better demand for this exposure."
    elif implication < -0.1:
        news_reason += f" Event implications are negative ({implication:+.1f}), so this was capped/trimmed despite any price strength."

    if inst.commodity in {"gold", "silver", "platinum", "palladium"}:
        macro_reason = "Macro: precious metals are useful when inflation, rates, or market stress matter."
    elif inst.commodity in {"wti_crude", "brent_crude"}:
        macro_reason = "Macro: crude oil is sensitive to OPEC, inventories, sanctions, shipping routes, and global growth."
    elif inst.commodity == "heating_oil_jet_fuel_proxy":
        macro_reason = "Macro: heating oil is the closest liquid proxy for diesel/jet fuel stress, so refinery disruption and aviation fuel shortages matter."
    elif inst.commodity == "gasoline":
        macro_reason = "Macro: gasoline is driven by refinery runs, driving demand, stockpiles, and seasonal cracks."
    elif inst.commodity == "natural_gas":
        macro_reason = "Macro: natural gas is sensitive to storage, LNG flows, weather, and European/US demand."
    elif inst.commodity in {"copper", "copper_miners"}:
        macro_reason = "Macro: copper is tied to China demand, industry, infrastructure, and risk appetite."
    elif inst.commodity in {"agriculture_basket", "corn", "wheat", "soybeans", "coffee", "cocoa", "sugar", "cotton", "live_cattle", "lean_hogs", "orange_juice"}:
        macro_reason = "Macro: agriculture and soft commodities are driven by weather, harvests, disease, export restrictions, and food inflation."
    else:
        macro_reason = "Macro: broad commodities diversify across inflation-sensitive assets."
    if not math.isnan(spy) and abs(spy) > 0.03:
        macro_reason += f" Equity market 1-month momentum is {spy*100:.1f}%, which affects risk appetite."

    risk = "high" if vol > 0.35 or confidence < 0.45 else ("medium" if vol > 0.20 or confidence < 0.65 else "lower")
    pct_reason = f"Target percentage: {(weight*100):.1f}% because its combined score was {score:.2f} with {confidence*100:.0f}% confidence."
    if cash_weight > 0:
        pct_reason += f" The rest keeps {(cash_weight*100):.1f}% outside this commodities pie as cash because signals were not strong enough for full deployment."

    return " ".join([
        price_reason,
        news_reason,
        macro_reason,
        f"Risk/confidence: {risk} risk, confidence {confidence*100:.0f}%.",
        pct_reason,
        f"Trading 212: search for {inst.trading212_name} ({inst.trading212_ticker}).",
    ])


def _allocate_from_scores(latest: pd.DataFrame, ml_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    df = latest.copy()
    df["ticker"] = df["ticker"].astype(str).str.lower()
    df = df[df["ticker"].isin(COMMODITY_UNIVERSE)].copy()
    selectable = df[~df["ticker"].isin(NON_SELECTABLE_SIGNAL_TICKERS)].copy()
    if df.empty:
        raise ValueError("No commodity feature rows available for allocation")

    for col in ["sent_mean_7d", "sent_shock", "event_score", "news_count_7d"]:
        if col not in df.columns:
            df[col] = 0.0

    # Futures can jump when contracts roll or when Yahoo stitches series. Bound
    # raw price features before cross-sectional scoring so one data artefact does
    # not crowd out the whole pie.
    mom_3 = _bounded_numeric(_column_or_default(df, "mom_3_1"), -0.50, 0.75)
    mom_1 = _bounded_numeric(_column_or_default(df, "mom_1_0"), -0.30, 0.50)
    sharpe_63 = _bounded_numeric(_column_or_default(df, "sharpe_63"), -3.0, 3.0)
    ma_200 = _bounded_numeric(_column_or_default(df, "ma_200_ratio", 1.0), 0.70, 1.50, default=1.0)
    vol_63 = _bounded_numeric(_column_or_default(df, "vol_63"), 0.0, 0.08)
    maxdd_63 = _bounded_numeric(_column_or_default(df, "maxdd_63"), -0.60, 0.0)
    roll_yield = _bounded_numeric(_column_or_default(df, "roll_yield"), -1.0, 1.0)
    price_score = (
        0.35 * _z(mom_3)
        + 0.25 * _z(mom_1)
        + 0.20 * _z(sharpe_63)
        + 0.15 * _z(ma_200)
        + 0.12 * _z(roll_yield)
        - 0.12 * _z(vol_63)
        - 0.10 * _z(-maxdd_63)
    )
    implication = _bounded_numeric(_column_or_default(df, "implication_score_7d"), -8.0, 8.0)
    implication_30 = _bounded_numeric(_column_or_default(df, "implication_score_30d"), -15.0, 15.0)
    news_score = (
        0.42 * _z(df["sent_mean_7d"])
        + 0.18 * _z(df["sent_shock"])
        + 0.15 * _z(df["event_score"])
        + 0.20 * _z(implication)
        + 0.05 * _z(implication_30)
    )
    fallback_score = 0.72 * price_score + 0.28 * news_score
    df["score"] = fallback_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["score_source"] = "zscore_fallback"

    if ml_scores is not None and not ml_scores.empty:
        df = df.merge(ml_scores, on="ticker", how="left")
        df["score"] = 0.65 * _z(df["ml_score"].fillna(0.0)) + 0.35 * df["score"]
        df["score_source"] = "lightgbm_blend"

    df["commodity"] = df["ticker"].map(lambda t: _instrument(t).commodity)
    df = df.sort_values(["score", "ticker"], ascending=[False, True])
    selectable = df[~df["ticker"].isin(NON_SELECTABLE_SIGNAL_TICKERS)].copy()

    # Keep one ETF/proxy per commodity bucket so the pie does not double-own the same exposure.
    chosen_rows = []
    used_commodities = set()
    for _, row in selectable.iterrows():
        commodity = row["commodity"]
        if commodity in used_commodities:
            continue
        chosen_rows.append(row)
        used_commodities.add(commodity)
        if len(chosen_rows) >= MAX_POSITIONS:
            break

    if len(chosen_rows) < MIN_POSITIONS:
        for _, row in selectable.iterrows():
            if str(row["ticker"]).lower() in {str(r["ticker"]).lower() for r in chosen_rows}:
                continue
            chosen_rows.append(row)
            if len(chosen_rows) >= MIN_POSITIONS:
                break

    chosen = pd.DataFrame(chosen_rows).head(MAX_POSITIONS).copy()
    chosen["positive_score"] = np.maximum(chosen["score"].astype(float) - float(chosen["score"].min()) + 0.15, 0.05)
    raw = chosen["positive_score"].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        raw = np.ones(len(chosen), dtype=float)
    # Vol-adjust so high-vol commodities (e.g. natural gas) do not take the
    # same notional risk as low-vol ones (e.g. gold).
    vol_63 = pd.to_numeric(_column_or_default(chosen, "vol_63"), errors="coerce").fillna(0.0)
    ann_vol = (vol_63 * math.sqrt(252)).clip(lower=0.05, upper=0.80)
    raw = raw * (1.0 / ann_vol.to_numpy(dtype=float))
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        raw = np.ones(len(chosen), dtype=float)
    base = raw / raw.sum()

    avg_score = float(chosen["score"].mean())
    avg_conf_seed = 1.0 / (1.0 + math.exp(-avg_score))
    avg_news = float(chosen["sent_mean_7d"].fillna(0.0).mean()) if "sent_mean_7d" in chosen else 0.0
    risk_penalty = float(pd.to_numeric(_column_or_default(chosen, "vol_63"), errors="coerce").fillna(0.0).mean() * math.sqrt(252))
    cash_weight = 0.0
    if avg_conf_seed < 0.47:
        cash_weight = 0.30
    elif avg_conf_seed < 0.54 or avg_news < -0.08:
        cash_weight = 0.20
    elif risk_penalty > 0.35:
        cash_weight = 0.10
    cash_weight = min(cash_weight, 1.0 - MIN_POSITIONS * MIN_COMMODITY_WEIGHT)

    investable = 1.0 - cash_weight
    weights = base * investable
    weights = np.clip(weights, MIN_COMMODITY_WEIGHT, MAX_COMMODITY_WEIGHT)
    weights = _renormalise_with_bounds(weights, investable, MIN_COMMODITY_WEIGHT, MAX_COMMODITY_WEIGHT)
    invested_weights = weights / max(investable, 1e-12)

    chosen["target_weight"] = weights
    chosen["invested_weight"] = invested_weights
    chosen["confidence"] = chosen["score"].map(lambda x: max(0.25, min(0.90, 1.0 / (1.0 + math.exp(-float(x))))))
    chosen["action"] = "BUY_OR_HOLD"
    chosen["trading212_name"] = chosen["ticker"].map(lambda t: _instrument(t).trading212_name)
    chosen["trading212_ticker"] = chosen["ticker"].map(lambda t: _instrument(t).trading212_ticker)
    chosen["reasons"] = chosen.apply(lambda r: _build_reasons(r, float(r["target_weight"]), cash_weight, float(r["score"]), float(r["confidence"])), axis=1)

    out = chosen[["date", "ticker", "commodity", "action", "score", "confidence", "target_weight", "invested_weight", "reasons", "trading212_name", "trading212_ticker"]].copy()
    if cash_weight > 1e-8:
        cash_reason = (
            f"Cash is kept outside the commodities Trading 212 pie. The commodity signal set was only moderate, "
            f"so {cash_weight*100:.1f}% is deliberately not deployed into the pie this run."
        )
        out = pd.concat([
            out,
            pd.DataFrame([{
                "date": out["date"].iloc[0],
                "ticker": CASH_TICKER,
                "commodity": "cash",
                "action": "HOLD_CASH",
                "score": 0.0,
                "confidence": 1.0,
                "target_weight": cash_weight,
                "invested_weight": 0.0,
                "reasons": cash_reason,
                "trading212_name": "Keep outside the pie as cash",
                "trading212_ticker": "CASH",
            }]),
        ], ignore_index=True)

    total = float(out["target_weight"].sum())
    if total > 0:
        out["target_weight"] = out["target_weight"] / total
    noncash = out["ticker"] != CASH_TICKER
    inv_total = float(out.loc[noncash, "target_weight"].sum())
    out.loc[noncash, "invested_weight"] = out.loc[noncash, "target_weight"] / max(inv_total, 1e-12)
    out.loc[~noncash, "invested_weight"] = 0.0
    out["asof_date"] = out["date"].apply(_iso_date)
    return out.drop(columns=["date"])


def _renormalise_with_bounds(weights: np.ndarray, target_sum: float, low: float, high: float) -> np.ndarray:
    w = np.array(weights, dtype=float)
    if len(w) == 0:
        return w
    target_sum = float(target_sum)
    for _ in range(50):
        diff = target_sum - float(w.sum())
        if abs(diff) < 1e-10:
            break
        if diff > 0:
            room = np.maximum(high - w, 0.0)
        else:
            room = np.maximum(w - low, 0.0)
        total_room = float(room.sum())
        if total_room <= 1e-12:
            break
        w += diff * (room / total_room)
        w = np.clip(w, low, high)
    if abs(float(w.sum()) - target_sum) > 1e-8 and w.sum() > 0:
        w = w * (target_sum / w.sum())
    return w


def make_commodity_recommendations(features: pd.DataFrame, news_features: pd.DataFrame | None = None) -> pd.DataFrame:
    feats = features.copy()
    feats["date"] = pd.to_datetime(feats["date"])
    if news_features is not None and not news_features.empty:
        nf = news_features.copy()
        nf["date"] = pd.to_datetime(nf["date"])
        merge_cols = [c for c in nf.columns if c not in {"commodity"}]
        feats = feats.merge(nf[merge_cols], on=["date", "ticker"], how="left")
    for col in [
        "news_count_7d", "news_count_30d", "sent_mean_7d", "sent_mean_30d", "sent_shock", "event_score",
        "implication_score_7d", "implication_score_30d", "implication_count_7d", "implication_count_30d",
    ]:
        if col not in feats.columns:
            feats[col] = 0.0
        feats[col] = feats[col].fillna(0.0)

    feats = add_commodity_roll_yield_features(feats)
    comm = feats[feats["ticker"].isin(COMMODITY_UNIVERSE)].copy()
    latest_date = comm["date"].max()

    # Historical monthly snapshots make the commodity backtest meaningful. News is
    # only merged where it exists for that exact date, so old snapshots remain
    # price/macro-only rather than accidentally using current headlines.
    dates = (
        comm[["date"]]
        .drop_duplicates()
        .sort_values("date")
        .assign(month=lambda x: x["date"].dt.to_period("M"))
        .groupby("month", as_index=False)["date"]
        .max()["date"]
        .tolist()
    )
    if latest_date not in dates:
        dates.append(latest_date)
    dates = sorted(set(pd.to_datetime(d) for d in dates))

    latest_ml = _lightgbm_scores(feats)
    out = []
    for d in dates:
        snap = comm[comm["date"] == d].copy()
        if snap["ticker"].nunique() < MIN_POSITIONS:
            continue
        # Let the first year of data warm up so long-horizon momentum is not all noise.
        if d < comm["date"].min() + pd.Timedelta(days=252):
            continue
        use_ml = latest_ml if d == latest_date else None
        try:
            out.append(_allocate_from_scores(snap, ml_scores=use_ml))
        except Exception:
            continue

    if not out:
        latest = comm[comm["date"] == latest_date].copy()
        return _allocate_from_scores(latest, ml_scores=latest_ml)
    return pd.concat(out, ignore_index=True)


def save_commodity_recommendations(recs: pd.DataFrame) -> int:
    if recs.empty:
        return 0
    recs = recs.copy()
    recs["asof_date"] = recs["asof_date"].astype(str)
    recs["ticker"] = recs["ticker"].astype(str).str.lower()
    total = recs.groupby("asof_date")["target_weight"].sum()
    bad = total[(total < 0.999) | (total > 1.001)]
    if not bad.empty:
        raise ValueError("Commodity recommendations must sum to 1.0 including cash: " + bad.to_string())
    noncash = recs[recs["ticker"] != CASH_TICKER]
    if not noncash.empty:
        if (noncash["target_weight"] < MIN_COMMODITY_WEIGHT - 1e-8).any():
            raise ValueError("Commodity non-cash recommendation below 10%")
        if (noncash["target_weight"] > MAX_COMMODITY_WEIGHT + 1e-8).any():
            raise ValueError("Commodity recommendation above 35%")
        counts = noncash.groupby("asof_date")["ticker"].nunique()
        bad_counts = counts[(counts < MIN_POSITIONS) | (counts > MAX_POSITIONS)]
        if not bad_counts.empty:
            raise ValueError("Commodity recommendations must have 3-5 non-cash holdings per snapshot: " + bad_counts.head().to_string())

    rows = [
        (
            r.asof_date,
            r.ticker,
            r.commodity,
            r.action,
            float(r.score),
            float(r.confidence),
            float(r.target_weight),
            float(r.invested_weight),
            r.reasons,
            r.trading212_name,
            r.trading212_ticker,
        )
        for r in recs.itertuples(index=False)
    ]
    with get_conn() as conn:
        asofs = recs["asof_date"].drop_duplicates().tolist()
        for asof in asofs:
            conn.execute("DELETE FROM commodity_recommendations WHERE asof_date = ?", (asof,))
        conn.executemany(
            """
            INSERT INTO commodity_recommendations
              (asof_date, ticker, commodity, action, score, confidence, target_weight, invested_weight, reasons, trading212_name, trading212_ticker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def save_commodity_holdings_and_trades(portfolio_value: Optional[float] = None) -> None:
    trade_value = float(portfolio_value if portfolio_value is not None else PORTFOLIO_VALUE)
    price_map = _latest_price_map()
    with get_conn() as conn:
        row = conn.execute("SELECT MAX(asof_date) FROM commodity_recommendations").fetchone()
        if not row or row[0] is None:
            return
        asof = row[0]
        recs = pd.read_sql_query(
            "SELECT * FROM commodity_recommendations WHERE asof_date = ? AND target_weight > 0 ORDER BY target_weight DESC",
            conn,
            params=(asof,),
        )
        prev_row = conn.execute("SELECT MAX(asof_date) FROM commodity_model_holdings WHERE asof_date < ?", (asof,)).fetchone()
        prev_asof = prev_row[0] if prev_row and prev_row[0] else None
        prev_shares = {}
        if prev_asof:
            prev = pd.read_sql_query("SELECT ticker, shares FROM commodity_model_holdings WHERE asof_date = ?", conn, params=(prev_asof,))
            prev_shares = dict(zip(prev["ticker"].astype(str), prev["shares"].astype(float)))

        holdings_rows = []
        for r in recs.itertuples(index=False):
            ticker = str(r.ticker).lower()
            px = float(price_map.get(ticker, 1.0 if ticker == CASH_TICKER else 0.0))
            if px <= 0:
                continue
            value = float(r.target_weight) * trade_value
            shares = 0.0 if ticker == CASH_TICKER else value / px
            holdings_rows.append((asof, ticker, r.commodity, float(r.target_weight), float(r.invested_weight), px, shares, value, r.trading212_name, r.trading212_ticker, r.reasons, float(r.confidence)))

        conn.execute("DELETE FROM commodity_model_holdings WHERE asof_date = ?", (asof,))
        conn.executemany(
            """
            INSERT INTO commodity_model_holdings
              (asof_date, ticker, commodity, target_weight, invested_weight, price, shares, value, trading212_name, trading212_ticker, reasons, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            holdings_rows,
        )

        trades = []
        current = {str(r.ticker).lower(): float(r.target_weight) for r in recs.itertuples(index=False)}
        all_tickers = set(current) | set(prev_shares)
        for ticker in all_tickers:
            if ticker == CASH_TICKER:
                continue
            px = float(price_map.get(ticker, 0.0))
            if px <= 0:
                continue
            new_shares = (current.get(ticker, 0.0) * trade_value) / px
            old_shares = float(prev_shares.get(ticker, 0.0))
            delta = new_shares - old_shares
            notional = abs(delta) * px
            if notional < MIN_TRADE_GBP:
                continue
            trades.append((asof, ticker, "BUY" if delta > 0 else "SELL", float(delta), float(notional)))

        conn.execute("DELETE FROM commodity_model_trades WHERE asof_date = ?", (asof,))
        if trades:
            conn.executemany(
                """
                INSERT INTO commodity_model_trades (asof_date, ticker, trade_action, shares_delta, est_notional)
                VALUES (?, ?, ?, ?, ?)
                """,
                trades,
            )
    print(f"[OK] Commodity Scout: saved {len(holdings_rows)} holdings and {len(trades)} trades for £{trade_value:,.2f}")


def run_commodity_backtest(cost_bps: float = 5.0, start_date: str | None = "2015-01-01") -> tuple[pd.DataFrame, dict]:
    with get_conn() as conn:
        params = list(COMMODITY_UNIVERSE)
        price_date_filter = ""
        rec_date_filter = ""
        if start_date:
            price_date_filter = " AND date >= ?"
            rec_date_filter = " WHERE asof_date >= ?"
            params.append(start_date)
        prices = pd.read_sql_query(
            """
            SELECT date, ticker, close
            FROM prices_daily
            WHERE ticker IN (%s)
            %s
            ORDER BY date ASC
            """ % (",".join(["?"] * len(COMMODITY_UNIVERSE)), price_date_filter),
            conn,
            params=params,
        )
        recs = pd.read_sql_query(
            """
            SELECT asof_date, ticker, target_weight
            FROM commodity_recommendations
            %s
            ORDER BY asof_date ASC
            """ % rec_date_filter,
            conn,
            params=(start_date,) if start_date else None,
        )
    curve, stats, _ = run_recommendation_backtest(prices, recs, cost_bps=cost_bps)
    if start_date:
        stats["start_date"] = start_date
        stats["period_label"] = f"Since {start_date[:4]}"
    return curve, stats


def run_commodity_pipeline(portfolio_value: Optional[float] = None, refresh_prices: bool = True, refresh_news: bool = True) -> None:
    init_db()
    if refresh_prices:
        update_commodity_prices(include_macro=False)

    prices = load_prices_for_commodities(include_macro=True)
    if prices.empty:
        print("[ERROR] Commodity Scout: no price data found. Run the main pipeline or refresh commodity prices first.")
        return
    current_asof_date = _iso_date(prices.loc[prices["ticker"].isin(COMMODITY_UNIVERSE), "date"].max())

    if refresh_news:
        fetch_and_store_commodity_news(days_back=30)
    news_features = build_commodity_news_features(asof_date=current_asof_date)
    if refresh_news:
        try:
            implication_features = update_implication_news(asof_date=current_asof_date, extra_tickers=COMMODITY_UNIVERSE)
            print(f"[OK] Commodity implication news: {len(implication_features)} directional scores")
        except Exception as e:
            print(f"[WARN] Commodity implication news failed (continuing without): {e}")
            implication_features = build_implication_features(asof_date=current_asof_date)
    else:
        implication_features = build_implication_features(asof_date=current_asof_date)
        if implication_features is not None and not implication_features.empty:
            print(f"[OK] Commodity implication news: reused {len(implication_features)} directional scores")
    if implication_features is not None and not implication_features.empty:
        news_features = news_features.merge(
            implication_features,
            on=["date", "ticker"],
            how="left",
        )
        for col in ["implication_score_7d", "implication_score_30d", "implication_count_7d", "implication_count_30d"]:
            if col in news_features.columns:
                news_features[col] = news_features[col].fillna(0.0)

    feats = build_feature_frame(prices)
    feats = add_cross_sectional_zscores(feats)
    feats = add_sector_relative_features(feats)

    recs = make_commodity_recommendations(feats, news_features)
    n = save_commodity_recommendations(recs)
    print(f"[OK] Commodity Scout: saved {n} recommendation rows")

    resolved_value = _resolve_portfolio_value(portfolio_value)
    save_commodity_holdings_and_trades(portfolio_value=resolved_value)

    latest_asof = str(recs["asof_date"].max())
    latest_recs = recs[recs["asof_date"].astype(str) == latest_asof]
    total = float(latest_recs["target_weight"].sum())
    invested = latest_recs[latest_recs["ticker"] != CASH_TICKER]
    inv_total = float(invested["invested_weight"].sum()) if not invested.empty else 0.0
    print(f"[OK] Commodity Scout latest target total: {total*100:.1f}% including cash")
    print(f"[OK] Trading 212 latest invested pie total: {inv_total*100:.1f}% excluding cash")
