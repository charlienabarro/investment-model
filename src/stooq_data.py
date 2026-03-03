# src/stooq_data.py
from __future__ import annotations

import io
import time
from typing import Optional

import pandas as pd
import requests


_BASE = "https://stooq.com/q/d/l/"


def _looks_like_no_data(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if "no data" in t[:200]:
        return True
    return False


def fetch_stooq_daily(ticker: str, retries: int = 5, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Stooq.

    Returns columns: date, open, high, low, close, volume
    Empty DF means no data.
    """
    sym = (ticker or "").strip().lower()
    if not sym:
        return pd.DataFrame()

    params = {"s": sym, "i": "d"}

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(_BASE, params=params, timeout=timeout, headers={"User-Agent": "investing-model/1.0"})
            if r.status_code != 200:
                time.sleep(1.0 + attempt * 0.75)
                continue

            if _looks_like_no_data(r.text):
                return pd.DataFrame()

            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                return pd.DataFrame()

            # Stooq format is typically: Date,Open,High,Low,Close,Volume
            cols = [c.strip().lower() for c in df.columns]
            df.columns = cols

            if "date" not in df.columns:
                # Sometimes "Date" becomes first unnamed column in odd responses
                return pd.DataFrame()

            keep = ["date", "open", "high", "low", "close", "volume"]
            for c in keep:
                if c not in df.columns:
                    df[c] = pd.NA

            out = df[keep].copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.dropna(subset=["date"])
            out = out.sort_values("date").reset_index(drop=True)

            for c in ["open", "high", "low", "close", "volume"]:
                out[c] = pd.to_numeric(out[c], errors="coerce")

            if out.empty:
                return pd.DataFrame()

            return out

        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt * 0.75)

    # Final fallback: fail closed
    return pd.DataFrame()