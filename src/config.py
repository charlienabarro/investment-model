# src/config.py — V2 realistic £350 portfolio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "market.db"

# ── Rebalance ──────────────────────────────────────────────
REBALANCE_FREQ = "M"

# ── Portfolio construction ─────────────────────────────────
# 8 holdings max: 6 equity, 1 bond ETF, 1 commodity ETF
EQUITY_K = 8
BONDS_K = 2
COMMS_K = 2
TOTAL_K = EQUITY_K + BONDS_K + COMMS_K

MAX_WEIGHT = 0.20        # 20% cap per name (8 names → room for concentration)
MIN_WEIGHT = 0.05        # 5% floor (£17.50 on £350 — above T212 min)

# ── Volatility targeting ───────────────────────────────────
TARGET_VOL = 0.10        # 10% annualised portfolio vol target
VOL_LOOKBACK = 63        # 3-month realised vol for scaling
VOL_FLOOR = 0.60         # never scale equity below 60% of raw weight
VOL_CAP = 1.20           # never scale equity above 120% of raw weight

# ── Drawdown circuit breaker (soft / graduated) ───────────
# As trailing drawdown deepens, shift equity → bonds gradually
# At DD_TILT_START, begin shifting; at DD_TILT_MAX, maximum shift applied
DD_TILT_START = -0.05    # start reducing equity at -5% trailing DD
DD_TILT_MAX = -0.12      # max reduction at -12% trailing DD
DD_MAX_SHIFT = 0.20      # shift up to 20% from equity to bonds
# Equity never goes below 50% (you want to stay invested)
EQUITY_FLOOR = 0.50

# ── Correlation filter ─────────────────────────────────────
CORR_WINDOW_DAYS = 252
CORR_THRESHOLD = 0.75

# ── Asset-class budgets (baseline, before vol/DD adjustments)
BUDGETS_BASE = {"equity": 0.70, "bonds": 0.18, "commodities": 0.12}

# ── ML signal ──────────────────────────────────────────────
ML_TRAIN_WINDOW_MONTHS = 36   # rolling 3-year training window (used by longest ensemble)
ML_MIN_TRAIN_SAMPLES = 200    # don't train if fewer rows than this
ML_FALLBACK_TO_ZSCORE = True  # if ML can't train, fall back to V1 z-score
ML_ENSEMBLE_WINDOWS = [12, 24, 36, 60]  # months: 1yr, 2yr, 3yr, 5yr lookback windows

# ── Realistic cost model (Trading 212 GBP account) ────────
PORTFOLIO_VALUE = 350.0
FX_FEE = 0.0015          # 0.15% on every USD trade
SPREAD_BPS = 0.5          # estimated 0.5bps spread
MIN_POSITION_GBP = 17.0   # don't create positions smaller than this
MIN_TRADE_GBP = 10.0      # ignore rebalances under £10 notional
NO_TRADE_BAND = 0.03      # 3% weight band — don't trade if delta < this
MAX_TURNOVER = 0.30        # cap one-way turnover at 30% per rebalance