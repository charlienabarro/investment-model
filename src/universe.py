# src/universe.py
from __future__ import annotations

from pathlib import Path
from typing import List
import re
import pandas as pd

from .config import BASE_DIR

CACHE_PATH = BASE_DIR / "data" / "universe_cache.csv"


# Extra diversifiers (all should exist on Stooq with .us)
# Note: these are ETFs, which is the easiest "commodities" route with free daily data.
CORE_ETFS = [
    # US equity
    "spy.us", "qqq.us", "dia.us", "iwm.us", "vti.us", "voo.us",

    # Bonds
    "ief.us", "tlt.us", "shy.us", "lqd.us", "hyg.us",

    # Gold / metals
    "gld.us", "iau.us", "slv.us",

    # Broad commodities + sub-sectors
    "dbc.us", "pdbc.us", "djp.us",  # broad commodities
    "uso.us", "ung.us",             # oil, gas (very volatile)
    "dba.us",                       # agriculture
    "cpER.us".lower().replace("cper.us", "cper.us"),  # copper ETF (keep literal)
    "copx.us",                      # copper miners proxy
]


def _ensure_data_dir() -> None:
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


def _to_stooq_us_ticker(t: str) -> str | None:
    """
    Convert tickers like 'BRK.B' or 'BF.B' into Stooq-ish 'brk.b.us' form.
    Wikipedia S&P500 uses dots for class shares.
    """
    t = t.strip().upper()
    if not t:
        return None

    # Keep only allowed characters: letters, numbers, dot
    if not re.match(r"^[A-Z0-9\.]+$", t):
        return None

    return f"{t.lower()}.us"


def _fetch_sp500_from_wikipedia() -> list[str]:
    import pandas as pd
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=25)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            "Failed to fetch S&P 500 list from Wikipedia (request blocked or network issue). "
            "Use cached universe or local universe file instead."
        ) from e

    html = resp.text

    # Guard: Wikipedia sometimes returns a block page or unexpected HTML
    if "List of S&P 500 companies" not in html or "constituents" not in html.lower():
        raise RuntimeError(
            "Wikipedia returned an unexpected page (possibly blocked). "
            "Delete universe_cache.csv and switch to local universe list."
        )

    try:
        tables = pd.read_html(html)
    except Exception as e:
        # IMPORTANT: do not include html in exception message
        raise RuntimeError(
            "Fetched Wikipedia page but failed to parse the S&P 500 table. "
            "This is usually due to bot-blocking or page structure changes."
        ) from e

    if not tables:
        raise RuntimeError("No tables found on Wikipedia S&P 500 page.")

    df = tables[0]
    sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    syms = df[sym_col].astype(str).tolist()

    out = []
    for s in syms:
        t = _to_stooq_us_ticker(s)
        if t:
            out.append(t)

    # de-dup preserve order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    return uniq


def build_universe(force_refresh: bool = False) -> List[str]:
    _ensure_data_dir()

    # 1) If cache exists, use it
    if CACHE_PATH.exists() and not force_refresh:
        df = pd.read_csv(CACHE_PATH)
        tickers = df["ticker"].astype(str).tolist()
        return tickers

    # 2) Always start with CORE_ETFS (includes commodities ETFs)
    merged = list(CORE_ETFS)

    # 3) Add lots of stocks from a local file you control
    extra_path = BASE_DIR / "data" / "universe_extra.csv"
    if extra_path.exists():
        extra = pd.read_csv(extra_path, header=None)[0].astype(str).str.strip().str.lower().tolist()
        merged.extend([t for t in extra if t])

    # 4) De-dup preserve order
    seen = set()
    out = []
    for t in merged:
        t = t.strip().lower()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)

    # 5) Write cache for fast future runs
    pd.DataFrame({"ticker": out}).to_csv(CACHE_PATH, index=False)
    return out


def get_universe() -> List[str]:
    """
    Main entry point used by pipeline/api.
    """
    return build_universe(force_refresh=False)