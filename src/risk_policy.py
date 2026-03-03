# src/risk_policy.py
from __future__ import annotations

# Minimal, robust sector + group mapping.
# You can expand SECTOR_MAP whenever you add tickers.

GROUP_MAP = {
    # Broad equity ETFs
    "spy.us": "us_equity",
    "qqq.us": "us_equity",
    "dia.us": "us_equity",
    "iwm.us": "us_equity",
    "vti.us": "us_equity",
    "voo.us": "us_equity",

    # Bonds / cashlike
    "shy.us": "bonds",
    "ief.us": "bonds",
    "tlt.us": "bonds",
    "lqd.us": "bonds",
    "hyg.us": "bonds",

    # Metals / commodities
    "gld.us": "gold",
    "iau.us": "gold",
    "slv.us": "metals",
    "dbc.us": "commodities",
    "pdbc.us": "commodities",
    "djp.us": "commodities",
    "copx.us": "metals",
}

DEFAULT_GROUP = "us_equity"

# Simple sector caps used by correlation filter selection.
MAX_PER_SECTOR = 2

SECTOR_MAP = {
    # Semis / hardware
    "asml.us": "Semiconductors",
    "nvda.us": "Semiconductors",
    "amd.us": "Semiconductors",
    "intc.us": "Semiconductors",
    "mu.us": "Semiconductors",
    "wdc.us": "Storage",
    "stx.us": "Storage",
    "klac.us": "Semiconductors",
    "lrcx.us": "Semiconductors",
    "amat.us": "Semiconductors",

    # Healthcare
    "jnj.us": "Healthcare",
    "mrk.us": "Healthcare",
    "abbv.us": "Healthcare",
    "pfe.us": "Healthcare",

    # Consumer
    "pg.us": "Consumer",
    "wmt.us": "Consumer",
    "ko.us": "Consumer",
    "pep.us": "Consumer",
    "mcd.us": "Consumer",

    # Energy
    "xom.us": "Energy",
    "cvx.us": "Energy",
    "hal.us": "Energy",
    "slb.us": "Energy",
    "oxy.us": "Energy",

    # Industrials
    "cat.us": "Industrials",
    "fdx.us": "Industrials",
    "ups.us": "Industrials",
    "de.us": "Industrials",

    # Tech / software
    "msft.us": "Software",
    "aapl.us": "Hardware",
    "amzn.us": "Internet",
    "googl.us": "Internet",
    "meta.us": "Internet",
    "adbe.us": "Software",
    "crm.us": "Software",
    "csco.us": "Networking",

    # Financials
    "jpm.us": "Financials",
    "v.us": "Financials",
    "ma.us": "Financials",
    "gs.us": "Financials",
    "ms.us": "Financials",
}


def get_sector(ticker: str) -> str:
    t = (ticker or "").lower()
    if t in SECTOR_MAP:
        return SECTOR_MAP[t]
    if t in GROUP_MAP:
        # ETFs grouped as their own "sector" for cap purposes
        return GROUP_MAP[t]
    if t == "cash":
        return "cash"
    return "Other"