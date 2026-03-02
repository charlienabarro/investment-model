# src/risk_policy.py — V2: added GICS-like sector mapping + sector caps

GROUP_MAP = {
    # Core ETFs
    "spy.us": "us_equity",
    "qqq.us": "us_equity",
    "dia.us": "us_equity",
    "iwm.us": "us_equity",
    "vti.us": "us_equity",
    "voo.us": "us_equity",

    "ief.us": "bonds",
    "tlt.us": "bonds",
    "shy.us": "bonds",
    "lqd.us": "bonds",
    "hyg.us": "bonds",

    "gld.us": "gold",
    "iau.us": "gold",
    "slv.us": "metals",

    # Commodities
    "dbc.us": "commodities",
    "pdbc.us": "commodities",
    "djp.us": "commodities",
    "uso.us": "energy",
    "ung.us": "energy",
    "dba.us": "agriculture",
    "cper.us": "metals",
    "copx.us": "metals",
}

DEFAULT_GROUP = "us_equity"  # individual stocks default to us_equity


# Hard caps by group (these are safety rails)
GROUP_CAPS = {
    "us_equity": 0.75,
    "em_equity": 0.20,
    "bonds": 0.80,
    "commodities": 0.35,
    "energy": 0.20,
    "agriculture": 0.20,
    "gold": 0.30,
    "metals": 0.25,
}


# ═══════════════════════════════════════════════════════════
# GICS-like sector mapping for individual stocks
# ═══════════════════════════════════════════════════════════
# This maps tickers to broad sectors. Used for:
# 1. Sector cap (max 2 equity names per sector)
# 2. Sector-relative features in features.py
#
# For tickers not listed here, we try to infer from the name,
# or default to "other". ETFs get their own sectors.

SECTOR_BY_TICKER = {
    # ── Technology ──
    "aapl.us": "tech", "msft.us": "tech", "googl.us": "tech", "goog.us": "tech",
    "meta.us": "tech", "amzn.us": "tech", "nflx.us": "tech", "crm.us": "tech",
    "adbe.us": "tech", "orcl.us": "tech", "csco.us": "tech", "intc.us": "tech",
    "ibm.us": "tech", "now.us": "tech", "uber.us": "tech", "abnb.us": "tech",
    "snap.us": "tech", "pins.us": "tech", "sq.us": "tech", "shop.us": "tech",
    "pltr.us": "tech", "net.us": "tech", "ddog.us": "tech", "snow.us": "tech",
    "crwd.us": "tech", "zs.us": "tech", "panw.us": "tech", "ftnt.us": "tech",

    # ── Semiconductors ──
    "nvda.us": "semis", "amd.us": "semis", "avgo.us": "semis", "qcom.us": "semis",
    "txn.us": "semis", "amat.us": "semis", "lrcx.us": "semis", "klac.us": "semis",
    "mrvl.us": "semis", "nxpi.us": "semis", "on.us": "semis", "adi.us": "semis",
    "mchp.us": "semis", "mu.us": "semis", "arm.us": "semis", "smci.us": "semis",
    "asml.us": "semis", "tsm.us": "semis",

    # ── Healthcare ──
    "unh.us": "health", "jnj.us": "health", "lly.us": "health", "pfe.us": "health",
    "abbv.us": "health", "mrk.us": "health", "tmo.us": "health", "abt.us": "health",
    "dhr.us": "health", "bmy.us": "health", "amgn.us": "health", "gild.us": "health",
    "isrg.us": "health", "vrtx.us": "health", "regn.us": "health", "mdlz.us": "health",
    "ci.us": "health", "hum.us": "health", "hca.us": "health", "elv.us": "health",
    "syk.us": "health", "bsx.us": "health", "eog.us": "health",

    # ── Financials ──
    "jpm.us": "financials", "bac.us": "financials", "wfc.us": "financials",
    "gs.us": "financials", "ms.us": "financials", "c.us": "financials",
    "blk.us": "financials", "schw.us": "financials", "spgi.us": "financials",
    "ice.us": "financials", "cme.us": "financials", "aon.us": "financials",
    "mmc.us": "financials", "pgr.us": "financials", "met.us": "financials",
    "aig.us": "financials", "brk.b.us": "financials", "v.us": "financials",
    "ma.us": "financials", "axp.us": "financials", "pypl.us": "financials",

    # ── Industrials ──
    "cat.us": "industrials", "de.us": "industrials", "hon.us": "industrials",
    "ge.us": "industrials", "rtx.us": "industrials", "lmt.us": "industrials",
    "ba.us": "industrials", "noc.us": "industrials", "gd.us": "industrials",
    "ups.us": "industrials", "fdx.us": "industrials", "wr.us": "industrials",
    "etn.us": "industrials", "itt.us": "industrials", "pcar.us": "industrials",

    # ── Consumer / Retail ──
    "wmt.us": "consumer", "cost.us": "consumer", "hd.us": "consumer",
    "low.us": "consumer", "tgt.us": "consumer", "ko.us": "consumer",
    "pep.us": "consumer", "pg.us": "consumer", "cl.us": "consumer",
    "mnst.us": "consumer", "sbux.us": "consumer", "mcd.us": "consumer",
    "nike.us": "consumer", "tsla.us": "consumer",

    # ── Energy ──
    "xom.us": "energy_stock", "cvx.us": "energy_stock", "cop.us": "energy_stock",
    "slb.us": "energy_stock", "eog.us": "energy_stock", "pxd.us": "energy_stock",
    "oxy.us": "energy_stock", "vlo.us": "energy_stock", "mpc.us": "energy_stock",
    "psx.us": "energy_stock", "hal.us": "energy_stock",

    # ── Utilities / REITs ──
    "nee.us": "utilities", "duk.us": "utilities", "so.us": "utilities",
    "d.us": "utilities", "aep.us": "utilities", "exc.us": "utilities",
    "pld.us": "reits", "amt.us": "reits", "cci.us": "reits",
    "eqr.us": "reits", "o.us": "reits", "spg.us": "reits",

    # ── Storage / Hardware ──
    "wdc.us": "storage", "stt.us": "storage",
}

# Max equity names from the same sector
MAX_PER_SECTOR = 2

def get_sector(ticker: str) -> str:
    """Get the GICS-like sector for a ticker."""
    t = ticker.lower()
    if t in SECTOR_BY_TICKER:
        return SECTOR_BY_TICKER[t]
    # ETFs and commodities — use their group
    grp = GROUP_MAP.get(t, DEFAULT_GROUP)
    if grp in ("bonds", "gold", "metals", "commodities", "energy", "agriculture"):
        return grp
    if grp == "us_equity":
        return "equity_etf"  # separate sector so ETFs don't compete with stock sector caps
    return "other"

# Index ETFs that should be eligible for equity selection
# These act as a stable core — lower vol than individual stocks
INDEX_ETFS = {"spy.us", "qqq.us", "dia.us", "iwm.us", "vti.us", "voo.us"}