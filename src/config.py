# src/config.py — V4: tighter risk, lower vol, harder DD breaker
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "market.db"

# ── Rebalance ──────────────────────────────────────────────
REBALANCE_FREQ = "M"

# ── Portfolio construction ─────────────────────────────────
# 8 holdings: 4 individual stocks + 2 index ETFs + 1 bond + 1 commodity
# The 2 index ETFs (SPY/QQQ/etc) act as a stable equity core,
# while the 4 stock picks provide alpha on top.
EQUITY_K = 6        # 6 equity slots total (model picks best mix of stocks + ETFs)
BONDS_K = 1
COMMS_K = 1
TOTAL_K = EQUITY_K + BONDS_K + COMMS_K  # 8

MAX_WEIGHT = 0.20   # 20% cap per name
MIN_WEIGHT = 0.05   # 5% floor

# ── Volatility targeting (TIGHTER) ─────────────────────────
TARGET_VOL = 0.10   # 10% annualised portfolio vol target
VOL_LOOKBACK = 63
VOL_FLOOR = 0.40    # was 0.60 — allow scaling equity down to 40% of raw weight
VOL_CAP = 1.10      # was 1.20 — less upside scaling when calm

# ── Drawdown circuit breaker (MORE AGGRESSIVE) ────────────
DD_TILT_START = -0.04    # was -0.05 — start shifting earlier
DD_TILT_MAX = -0.10      # was -0.12 — hit maximum shift sooner
DD_MAX_SHIFT = 0.30      # was 0.20 — shift up to 30% from equity to bonds
EQUITY_FLOOR = 0.40      # was 0.50 — allow equity to drop to 40% in bad times

# ── Correlation filter ─────────────────────────────────────
CORR_WINDOW_DAYS = 252
CORR_THRESHOLD = 0.70    # was 0.75 — stricter correlation filter

# ── Asset-class budgets (baseline) ─────────────────────────
# Slightly more defensive baseline: less equity, more bonds
BUDGETS_BASE = {"equity": 0.65, "bonds": 0.22, "commodities": 0.13}

# ── ML signal ──────────────────────────────────────────────
ML_TRAIN_WINDOW_MONTHS = 36
ML_MIN_TRAIN_SAMPLES = 200
ML_FALLBACK_TO_ZSCORE = True
ML_ENSEMBLE_WINDOWS = [12, 24, 36, 60]

# ── Realistic cost model (Trading 212 GBP account) ────────
PORTFOLIO_VALUE = 350.0
FX_FEE = 0.0015
SPREAD_BPS = 0.5
MIN_POSITION_GBP = 17.0
MIN_TRADE_GBP = 10.0
NO_TRADE_BAND = 0.03
MAX_TURNOVER = 0.30