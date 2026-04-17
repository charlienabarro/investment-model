# src/config.py
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "prices.sqlite3"

# ----------------------------
# Universe (Stooq symbols)
# ----------------------------
# NOTE: Stooq uses class-share tickers with dash: brk-b.us not brk.b.us.
# We'll also normalise in stooq_data.py, but keep the universe clean.
DEFAULT_UNIVERSE: list[str] = [
    # Regime / macro anchors
    "spy.us",
    "shy.us",
    "ief.us",
    "tlt.us",
    "gld.us",
    "slv.us",
    "dbc.us",
    "qqq.us",
]

# ----------------------------
# Portfolio construction
# ----------------------------
# Selection counts (used by V4 model)
EQUITY_K = 6
BONDS_K = 2
COMMS_K = 1
TOTAL_K = EQUITY_K + BONDS_K + COMMS_K

# Weight constraints
MAX_WEIGHT = 0.22
MIN_WEIGHT = 0.03

# Correlation filter (equities)
CORR_WINDOW_DAYS = 126
CORR_THRESHOLD = 0.80

# Base budgets inside the risky sleeve (cash is handled separately in model.py)
BUDGETS_BASE = {"equity": 0.45, "bonds": 0.45, "commodities": 0.10}
EQUITY_FLOOR = 0.15  # never let equity go below this *within risky sleeve* unless cash takes over

# Vol targeting
TARGET_VOL = 0.08          # 8% annualised target for the risky sleeve
VOL_LOOKBACK = 63          # used as default; model will also look at 21d for responsiveness
VOL_FLOOR = 0.60
VOL_CAP = 1.20

# Drawdown tilt (SPY-based)
DD_TILT_START = -0.08      # start shifting from equity -> bonds if SPY drawdown worse than -8%
DD_TILT_MAX = -0.18        # max stress at -18%
DD_MAX_SHIFT = 0.35        # shift up to 35% from equity -> bonds inside risky sleeve

# ML training knobs
ML_MIN_TRAIN_SAMPLES = 250
ML_TRAIN_WINDOW_MONTHS = 60
ML_FALLBACK_TO_ZSCORE = True
ML_ENSEMBLE_WINDOWS = [24, 36, 60]  # months

# Cash sleeve
CASH_TICKER = "cash"
CASH_MAX_WEIGHT = 0.50

# Live trading / execution realism. This is only a fallback seed when there is no
# saved portfolio snapshot to mark to market yet.
PORTFOLIO_VALUE = float(os.environ.get("PORTFOLIO_VALUE", "10000.0"))
MIN_TRADE_GBP = 5.0
NO_TRADE_BAND = 0.01
MAX_TURNOVER = 0.35
FINAL_MIN_POSITION_WEIGHT = 0.09

# Backtest window
BACKTEST_START_DATE = "2005-01-01"

# Backtest realism knobs
DEFAULT_COST_BPS = 20.0
DEFAULT_SLIPPAGE_BPS = 10.0
MGMT_FEE_BPS_PER_YEAR = 50.0
