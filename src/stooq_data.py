# src/stooq_data.py
from __future__ import annotations

import io
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests


# Stooq has historically supported both stooq.pl and stooq.com.
# In practice, stooq.pl is the most reliable for the CSV download endpoint.
_BASE_URLS = [
    "https://stooq.pl/q/d/l/",
    "https://stooq.com/q/d/l/",
]


def _download_stooq_csv(symbol: str, timeout: float = 20.0) -> Tuple[Optional[str], Optional[str]]:
    """Return (CSV text, error detail)."""
    params = {"s": symbol, "i": "d"}
    headers = {
        # Some endpoints will return empty data without a UA.
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
    }
    errors = []

    for base in _BASE_URLS:
        host = urlparse(base).netloc or base
        try:
            r = requests.get(base, params=params, headers=headers, timeout=timeout)
            if r.status_code != 200:
                errors.append(f"{host} HTTP {r.status_code}")
                continue
            txt = (r.text or "").strip()
            if not txt:
                errors.append(f"{host} empty response")
                continue
            # Valid Stooq CSV starts with "Date,Open,High,Low,Close".
            # If we don't see a Date column at all, it's not real data.
            if "Date" not in txt.splitlines()[0]:
                errors.append(f"{host} unexpected header")
                continue
            return txt, None
        except requests.RequestException as e:
            errors.append(f"{host} {type(e).__name__}")
            continue

    return None, "; ".join(errors) if errors else "stooq request failed"


def _to_yahoo_symbol(ticker: str) -> str:
    """
    Convert project ticker format like 'brk.b.us' to Yahoo symbol 'BRK-B'.
    """
    t = (ticker or "").strip()
    if t.lower().endswith(".us"):
        t = t[:-3]
    return t.upper().replace(".", "-")


def _download_yahoo_chart(ticker: str, timeout: float = 20.0) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch daily OHLCV using Yahoo chart endpoint.
    """
    sym = _to_yahoo_symbol(ticker)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    params = {"interval": "1d", "range": "max"}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return pd.DataFrame(), f"yahoo HTTP {r.status_code}"

        payload = r.json()
        chart = (payload or {}).get("chart", {})
        result = (chart.get("result") or [])
        if not result:
            err = chart.get("error")
            if err:
                return pd.DataFrame(), f"yahoo error: {err}"
            return pd.DataFrame(), "yahoo empty result"

        block = result[0]
        ts = block.get("timestamp") or []
        quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
        if not ts or not quote:
            return pd.DataFrame(), "yahoo missing timestamp/quote"

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
                "open": pd.to_numeric(quote.get("open"), errors="coerce"),
                "high": pd.to_numeric(quote.get("high"), errors="coerce"),
                "low": pd.to_numeric(quote.get("low"), errors="coerce"),
                "close": pd.to_numeric(quote.get("close"), errors="coerce"),
                "volume": pd.to_numeric(quote.get("volume"), errors="coerce"),
            }
        )
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            return df, "yahoo parsed but no valid rows"
        return df, None
    except requests.RequestException as e:
        return pd.DataFrame(), f"yahoo {type(e).__name__}"
    except Exception as e:
        return pd.DataFrame(), f"yahoo parse error: {e}"


def fetch_daily_with_fallback(
    ticker: str, max_retries: int = 2, sleep_s: float = 0.5, request_timeout: float = 8.0
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """
    Fetch daily OHLCV from Stooq, fallback to Yahoo chart API.

    Input ticker format in the project is lower-case like "spy.us".
    Returns: (df, source, error_message_if_any)
    """
    t = (ticker or "").strip().lower()
    if not t:
        return pd.DataFrame(), "none", "empty ticker"

    # Try both cases – if Stooq changes/varies its symbol rules, this keeps us alive.
    candidates = [t, t.upper()]
    errors = []

    for attempt in range(max_retries):
        for sym in candidates:
            try:
                csv_txt, err = _download_stooq_csv(sym, timeout=request_timeout)
                if not csv_txt:
                    if err:
                        errors.append(err)
                    continue
                try:
                    df = pd.read_csv(io.StringIO(csv_txt))
                except Exception:
                    errors.append("stooq parse csv failed")
                    continue

                # Expected columns: Date, Open, High, Low, Close, Volume
                if "Date" not in df.columns or "Close" not in df.columns:
                    errors.append("stooq missing Date/Close columns")
                    continue

                df = df.rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                df = df.dropna(subset=["date", "close"]).copy()
                if df.empty:
                    continue

                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).copy()

                for c in ["open", "high", "low", "close", "volume"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                df = df.dropna(subset=["close"]).sort_values("date")
                if df.empty:
                    errors.append("stooq no valid rows after parsing")
                    continue

                return (
                    df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True),
                    "stooq",
                    None,
                )
            except KeyboardInterrupt:
                return pd.DataFrame(), "none", "interrupted while fetching stooq data"

        try:
            time.sleep(sleep_s * (attempt + 1))
        except KeyboardInterrupt:
            return pd.DataFrame(), "none", "interrupted during retry backoff"

    ydf, yerr = _download_yahoo_chart(t, timeout=request_timeout)
    if not ydf.empty:
        return ydf, "yahoo", None
    if yerr:
        errors.append(yerr)

    uniq_errors = list(dict.fromkeys([e for e in errors if e]))
    err_msg = "; ".join(uniq_errors[:3]) if uniq_errors else "no data from stooq/yahoo"
    return pd.DataFrame(), "none", err_msg


def fetch_stooq_daily(
    ticker: str, max_retries: int = 2, sleep_s: float = 0.5, request_timeout: float = 8.0
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.
    """
    df, _, _ = fetch_daily_with_fallback(
        ticker,
        max_retries=max_retries,
        sleep_s=sleep_s,
        request_timeout=request_timeout,
    )
    return df
