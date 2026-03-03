# src/analyst_events.py
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

UP_RE = re.compile(r"\b(upgrade|upgraded|raises|raised|boosts|boosted)\b", re.I)
DOWN_RE = re.compile(r"\b(downgrade|downgraded|cuts|cut|lowers|lowered)\b", re.I)
PT_RE = re.compile(r"\b(price\s*target|pt)\b", re.I)
BANK_RE = re.compile(r"\b(goldman|morgan\s*stanley|j\.?p\.?\s*morgan|barclays|ubs|bofa|bank\s+of\s+america|citi|deutsche|jefferies|wells\s*fargo)\b", re.I)

def build_analyst_event_features(
    articles_df,
    tickers: Iterable[str],
    asof_date: str,
) -> Dict[str, Dict[str, float]]:
    """
    articles_df must include: date, ticker, title, summary (or description).
    Returns per-ticker event counts over 7/30 days.
    """
    asof = datetime.fromisoformat(asof_date)
    d7 = asof - timedelta(days=7)
    d30 = asof - timedelta(days=30)

    out: Dict[str, Dict[str, float]] = {t: {
        "analyst_up_7d": 0.0, "analyst_down_7d": 0.0, "pt_mention_7d": 0.0, "broker_note_7d": 0.0,
        "analyst_up_30d": 0.0, "analyst_down_30d": 0.0, "pt_mention_30d": 0.0, "broker_note_30d": 0.0,
    } for t in tickers}

    if articles_df is None or len(articles_df) == 0:
        return out

    df = articles_df.copy()
    df["date"] = df["date"].astype(str)
    df["dt"] = df["date"].map(lambda s: datetime.fromisoformat(s[:10]) if len(s) >= 10 else None)
    df = df[df["dt"].notna()]
    df["ticker"] = df["ticker"].astype(str).str.lower()

    def txt(row):
        return f"{row.get('title','')} {row.get('summary','')} {row.get('description','')}".strip()

    for _, r in df.iterrows():
        t = r["ticker"]
        if t not in out:
            continue
        body = txt(r)
        dt = r["dt"]
        is_up = bool(UP_RE.search(body))
        is_down = bool(DOWN_RE.search(body))
        is_pt = bool(PT_RE.search(body))
        is_bank = bool(BANK_RE.search(body))

        if dt >= d7:
            out[t]["analyst_up_7d"] += 1.0 if is_up else 0.0
            out[t]["analyst_down_7d"] += 1.0 if is_down else 0.0
            out[t]["pt_mention_7d"] += 1.0 if is_pt else 0.0
            out[t]["broker_note_7d"] += 1.0 if is_bank else 0.0

        if dt >= d30:
            out[t]["analyst_up_30d"] += 1.0 if is_up else 0.0
            out[t]["analyst_down_30d"] += 1.0 if is_down else 0.0
            out[t]["pt_mention_30d"] += 1.0 if is_pt else 0.0
            out[t]["broker_note_30d"] += 1.0 if is_bank else 0.0

    return out