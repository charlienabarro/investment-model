# src/config.py
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "prices.db"

# ----------------------------
# Universe
# ----------------------------
DEFAULT_UNIVERSE: list[str] = [
    "spy.us",
    "qqq.us",
    "shy.us",
    "ief.us",
    "tlt.us",
    "gld.us",
    "slv.us",
    "dbc.us",
]

# ----------------------------
# Portfolio sizing
# ----------------------------
TOP_N = 8
EQUITY_K = 6
BONDS_K = 1
COMMS_K = 1
TOTAL_K = TOP_N

MAX_WEIGHT = 0.22
MIN_WEIGHT = 0.05

# ----------------------------
# Risk budgets (live)
# ----------------------------
BUDGETS_RISK_ON = {"equity": 0.55, "bonds": 0.35, "commodities": 0.10}
BUDGETS_RISK_OFF = {"equity": 0.35, "bonds": 0.55, "commodities": 0.10}

# Used by model as starting point
BUDGETS_BASE = dict(BUDGETS_RISK_ON)

# ----------------------------
# Correlation control
# ----------------------------
CORR_WINDOW_DAYS = 126
CORR_THRESHOLD = 0.80

# ----------------------------
# Volatility targeting
# ----------------------------
TARGET_VOL = 0.12
VOL_LOOKBACK = 63
VOL_FLOOR = 0.60
VOL_CAP = 1.30

# ----------------------------
# Drawdown shift
# ----------------------------
DD_TILT_START = -0.08
DD_TILT_MAX = -0.15
DD_MAX_SHIFT = 0.20
EQUITY_FLOOR = 0.35

# ----------------------------
# ML settings
# ----------------------------
ML_TRAIN_WINDOW_MONTHS = 60
ML_MIN_TRAIN_SAMPLES = 400
ML_FALLBACK_TO_ZSCORE = True
ML_ENSEMBLE_WINDOWS = [24, 48, 72]

# ----------------------------
# Trading and portfolio value
# ----------------------------
PORTFOLIO_VALUE = 350.0
MIN_TRADE_GBP = 5.0
NO_TRADE_BAND = 0.02
MAX_TURNOVER = 0.25

# Backtest realism knobs
DEFAULT_COST_BPS = 20.0
DEFAULT_SLIPPAGE_BPS = 10.0
MGMT_FEE_BPS_PER_YEAR = 50.0