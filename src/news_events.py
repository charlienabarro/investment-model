# src/news_events.py
from __future__ import annotations

import re
import time
from typing import Dict, List
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from .config import BASE_DIR

CACHE_DIR = BASE_DIR / "data" / "news_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "investment-model/1.0"}

POS_WORDS = {
    "beats", "beat", "surge", "soar", "record", "strong", "raises", "raise", "upgrade", "upgraded",
    "bullish", "outperform", "buy", "profit", "growth", "guidance raised"
}
NEG_WORDS = {
    "miss", "misses", "plunge", "drop", "weak", "cuts", "cut", "downgrade", "downgraded",
    "bearish", "underperform", "sell", "loss", "lawsuit", "probe", "warning", "guidance cut"
}

RE_UPGRADE = re.compile(r"\bupgrade(d)?\b|\braise(s|d)?\s+rating\b|\binitiated\s+at\s+buy\b", re.I)
RE_DOWNGRADE = re.compile(r"\bdowngrade(d)?\b|\bcut(s|)?\s+rating\b|\binitiated\s+at\s+sell\b", re.I)
RE_PT_UP = re.compile(r"\bprice\s+target\b.*\braise(d|s)?\b|\bpt\b.*\braise(d|s)?\b", re.I)
RE_PT_DOWN = re.compile(r"\bprice\s+target\b.*\bcut(s|)?\b|\bpt\b.*\bcut(s|)?\b", re.I)


def _http_get(url: str, sleep_s: float = 0.15) -> str:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        txt = resp.read().decode("utf-8", errors="ignore")
    time.sleep(sleep_s)
    return txt


def _ticker_to_yahoo_symbol(tkr: str) -> str:
    # aapl.us -> AAPL
    t = (tkr or "").lower().strip()
    if t.endswith(".us"):
        t = t[:-3]
    return t.upper()


def _fetch_yahoo_rss_items(ticker: str) -> pd.DataFrame:
    sym = _ticker_to_yahoo_symbol(ticker)
    cache_path = CACHE_DIR / f"{sym}.xml"

    # Cache for 30 mins
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 1800:
            xml = cache_path.read_text(encoding="utf-8", errors="ignore")
        else:
            xml = _http_get(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US")
            cache_path.write_text(xml, encoding="utf-8")
    else:
        xml = _http_get(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US")
        cache_path.write_text(xml, encoding="utf-8")

    try:
        root = ET.fromstring(xml)
    except Exception:
        return pd.DataFrame(columns=["date", "title"])

    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        # Parse date loosely by pandas
        dt = pd.to_datetime(pub, errors="coerce", utc=True)
        if pd.isna(dt):
            continue
        items.append({"date": dt.tz_convert(None).normalize(), "title": title})

    df = pd.DataFrame(items)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["date", "title"])
    return df


def _headline_sentiment_score(title: str) -> float:
    t = (title or "").lower()
    score = 0.0
    for w in POS_WORDS:
        if w in t:
            score += 1.0
    for w in NEG_WORDS:
        if w in t:
            score -= 1.0
    return score


def build_news_features(tickers: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Daily features (per ticker):
      sent_mean_7d, sent_mean_30d, sent_shock, news_count_7d
      upgrades_30d, downgrades_30d, pt_raises_30d, pt_cuts_30d
    """
    if not tickers:
        return pd.DataFrame()

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()
    if end_date < start_date:
        return pd.DataFrame()

    all_rows = []
    for tkr in tickers:
        items = _fetch_yahoo_rss_items(tkr)
        cal = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
        cal["ticker"] = tkr

        if items.empty:
            # no news -> zeros
            for c in ["sent_mean_7d", "sent_mean_30d", "sent_shock", "news_count_7d",
                      "upgrades_30d", "downgrades_30d", "pt_raises_30d", "pt_cuts_30d"]:
                cal[c] = 0.0
            all_rows.append(cal)
            continue

        items["sent"] = items["title"].apply(_headline_sentiment_score)
        items["is_upgrade"] = items["title"].apply(lambda s: 1 if RE_UPGRADE.search(s or "") else 0)
        items["is_downgrade"] = items["title"].apply(lambda s: 1 if RE_DOWNGRADE.search(s or "") else 0)
        items["is_pt_up"] = items["title"].apply(lambda s: 1 if RE_PT_UP.search(s or "") else 0)
        items["is_pt_down"] = items["title"].apply(lambda s: 1 if RE_PT_DOWN.search(s or "") else 0)

        daily = items.groupby("date").agg(
            sent_mean=("sent", "mean"),
            news_count=("sent", "size"),
            upgrades=("is_upgrade", "sum"),
            downgrades=("is_downgrade", "sum"),
            pt_up=("is_pt_up", "sum"),
            pt_down=("is_pt_down", "sum"),
        ).reset_index()

        out = cal.merge(daily, on="date", how="left")
        out[["sent_mean", "news_count", "upgrades", "downgrades", "pt_up", "pt_down"]] = out[
            ["sent_mean", "news_count", "upgrades", "downgrades", "pt_up", "pt_down"]
        ].fillna(0.0)

        out["sent_mean_7d"] = out["sent_mean"].rolling(7, min_periods=1).mean()
        out["sent_mean_30d"] = out["sent_mean"].rolling(30, min_periods=1).mean()
        out["news_count_7d"] = out["news_count"].rolling(7, min_periods=1).sum()

        # shock = today sentiment minus 30d average (simple)
        out["sent_shock"] = out["sent_mean"] - out["sent_mean_30d"]

        out["upgrades_30d"] = out["upgrades"].rolling(30, min_periods=1).sum()
        out["downgrades_30d"] = out["downgrades"].rolling(30, min_periods=1).sum()
        out["pt_raises_30d"] = out["pt_up"].rolling(30, min_periods=1).sum()
        out["pt_cuts_30d"] = out["pt_down"].rolling(30, min_periods=1).sum()

        out = out.drop(columns=["sent_mean", "news_count", "upgrades", "downgrades", "pt_up", "pt_down"])
        all_rows.append(out)

    out = pd.concat(all_rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out