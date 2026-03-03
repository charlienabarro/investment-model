# src/sec_edgar.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from urllib.request import Request, urlopen
from urllib.error import URLError

from .config import BASE_DIR

SEC_CACHE_DIR = BASE_DIR / "data" / "sec_cache"
SEC_PROGRESS_PATH = BASE_DIR / "data" / "sec_progress.json"

SEC_USER_AGENT = "investing-model/1.0 (email: you@example.com)"  # <-- change to yours
SEC_SLEEP_S = 0.25  # be polite


def _ensure_dirs() -> None:
    SEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


def _http_get_json(url: str, timeout: int = 30) -> dict:
    req = Request(url, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        return json.loads(raw)


def _load_progress() -> dict:
    if not SEC_PROGRESS_PATH.exists():
        return {"done": [], "last_ticker": None, "updated_at": None}
    try:
        return json.loads(SEC_PROGRESS_PATH.read_text())
    except Exception:
        return {"done": [], "last_ticker": None, "updated_at": None}


def _save_progress(done: List[str], last_ticker: Optional[str]) -> None:
    payload = {
        "done": sorted(list(set(done))),
        "last_ticker": last_ticker,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    SEC_PROGRESS_PATH.write_text(json.dumps(payload, indent=2))


def _ticker_cache_path(ticker: str) -> Path:
    safe = ticker.replace(".", "_").lower()
    return SEC_CACHE_DIR / f"sec_features_{safe}.csv"


def _load_cached_ticker(ticker: str) -> Optional[pd.DataFrame]:
    p = _ticker_cache_path(ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None


def _save_cached_ticker(ticker: str, df: pd.DataFrame) -> None:
    p = _ticker_cache_path(ticker)
    df.to_csv(p, index=False)


def _cik10_from_ticker_guess(ticker: str) -> Optional[str]:
    """
    You likely already have a mapping elsewhere.
    If you DON'T, keep this returning None and only run SEC on tickers where you have CIKs.
    """
    return None


def _fetch_company_submissions(cik10: str) -> dict:
    # SEC wants zero-padded 10-digit CIK in the URL:
    # https://data.sec.gov/submissions/CIK0000320193.json
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    time.sleep(SEC_SLEEP_S)
    return _http_get_json(url)


def _build_features_from_submissions(ticker: str, subs: dict,
                                    start_date: pd.Timestamp,
                                    end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Produces monthly-ish, low-leakage SEC features:
      - filing_count_30d
      - filing_count_90d
      - has_10k_30d / has_10q_30d
      - days_since_last_filing
    """
    items = subs.get("filings", {}).get("recent", {})
    form = items.get("form", [])
    filing_date = items.get("filingDate", [])

    if not form or not filing_date:
        return pd.DataFrame(columns=["date", "ticker"])

    df = pd.DataFrame({"form": form, "filingDate": filing_date})
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    df = df.dropna(subset=["filingDate"])
    df = df.sort_values("filingDate")

    # filter window
    df = df[(df["filingDate"] >= start_date) & (df["filingDate"] <= end_date)]
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    # build daily index then later you can merge on month-end
    all_days = pd.date_range(start_date.normalize(), end_date.normalize(), freq="D")
    out = pd.DataFrame({"date": all_days})
    out["ticker"] = ticker

    # rolling counts
    df["one"] = 1
    df = df.set_index("filingDate")

    c30 = df["one"].rolling("30D").sum().reindex(all_days, method="ffill").fillna(0.0)
    c90 = df["one"].rolling("90D").sum().reindex(all_days, method="ffill").fillna(0.0)

    # 10-K / 10-Q flags in last 30D
    is_10k = (df["form"] == "10-K").astype(int).rolling("30D").max()
    is_10q = (df["form"] == "10-Q").astype(int).rolling("30D").max()
    has10k30 = is_10k.reindex(all_days, method="ffill").fillna(0).astype(int)
    has10q30 = is_10q.reindex(all_days, method="ffill").fillna(0).astype(int)

    # days since last filing
    last_filing = df.index.to_series().reindex(all_days, method="ffill")
    days_since = (pd.Series(all_days, index=all_days) - last_filing).dt.days
    days_since = days_since.fillna(9999).astype(int)

    out["sec_filings_30d"] = c30.values
    out["sec_filings_90d"] = c90.values
    out["sec_has_10k_30d"] = has10k30.values
    out["sec_has_10q_30d"] = has10q30.values
    out["sec_days_since_last"] = days_since.values

    return out


def build_sec_filing_features(tickers: List[str],
                              start_date: str | pd.Timestamp,
                              end_date: str | pd.Timestamp,
                              force_refresh: bool = False) -> pd.DataFrame:
    """
    Resumable: caches per-ticker features and progress.
    If a ticker fails (timeout/limit), it saves progress and returns what it has so far.
    """
    _ensure_dirs()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    prog = _load_progress()
    done = set([t.lower() for t in prog.get("done", [])])

    all_rows: List[pd.DataFrame] = []

    for t in [x.lower() for x in tickers]:
        # already cached + marked done
        if (t in done) and (not force_refresh):
            cached = _load_cached_ticker(t)
            if cached is not None and not cached.empty:
                all_rows.append(cached)
            continue

        # try cached even if not done (partial runs)
        if not force_refresh:
            cached = _load_cached_ticker(t)
            if cached is not None and not cached.empty:
                all_rows.append(cached)
                done.add(t)
                _save_progress(list(done), last_ticker=t)
                continue

        cik10 = _cik10_from_ticker_guess(t)
        if not cik10:
            # no mapping: skip but don't crash
            done.add(t)
            _save_progress(list(done), last_ticker=t)
            continue

        try:
            subs = _fetch_company_submissions(cik10)
            feat = _build_features_from_submissions(t, subs, start, end)
            _save_cached_ticker(t, feat)
            all_rows.append(feat)

            done.add(t)
            _save_progress(list(done), last_ticker=t)

        except URLError as e:
            # Save progress and return what we have; next run continues.
            _save_progress(list(done), last_ticker=t)
            print(f"[WARN] SEC fetch failed for {t} (will resume next run): {e}")
            break
        except Exception as e:
            _save_progress(list(done), last_ticker=t)
            print(f"[WARN] SEC error for {t} (will resume next run): {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.lower()
    return out