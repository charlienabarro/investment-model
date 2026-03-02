# src/model.py — V4: walk-forward validated multi-timeframe ensemble
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from .news_sentiment import update_news_sentiment
from .optimiser import optimise_long_only
from .config import (
    EQUITY_K, BONDS_K, COMMS_K, TOTAL_K,
    MAX_WEIGHT, MIN_WEIGHT,
    CORR_WINDOW_DAYS, CORR_THRESHOLD,
    BUDGETS_BASE,
    TARGET_VOL, VOL_LOOKBACK, VOL_FLOOR, VOL_CAP,
    DD_TILT_START, DD_TILT_MAX, DD_MAX_SHIFT, EQUITY_FLOOR,
    ML_TRAIN_WINDOW_MONTHS, ML_MIN_TRAIN_SAMPLES, ML_FALLBACK_TO_ZSCORE,
    ML_ENSEMBLE_WINDOWS,
)
from .risk_policy import GROUP_MAP, DEFAULT_GROUP, get_sector, MAX_PER_SECTOR

warnings.filterwarnings("ignore", category=UserWarning)

# ── ML feature columns (expanded V3) ──
ML_FEATURES = [
    "mom_12_1_z", "mom_6_1_z", "mom_3_1_z", "mom_1_0_z",
    "ma_200_ratio_z", "trend_50_200_z",
    "vol_63_z", "vol_252_z", "maxdd_252_z", "maxdd_63_z",
    "sharpe_63_z", "sharpe_252_z",
    "mr_zscore_21_z", "rsi_14_z",
    "skew_63_z", "gap_ratio_63_z", "vol_ratio_21_63_z",
    "rvol_20_z", "vpt_21_z", "obv_slope_21_z", "vol_vol_21_z",
    "spy_mom_1m_z", "spy_vol_63_z", "yield_curve_z", "credit_spread_z", "gold_trend_z",
    "mom_12_1_sec_rel", "mom_6_1_sec_rel", "mom_3_1_sec_rel",
    "sharpe_63_sec_rel", "vol_63_sec_rel", "ma_200_ratio_sec_rel",
]

FALLBACK_WEIGHTS = {
    "mom_12_1_z": 0.25, "mom_6_1_z": 0.15, "mom_3_1_z": 0.10,
    "ma_200_ratio_z": 0.15, "trend_50_200_z": 0.10,
    "vol_63_z": -0.15, "maxdd_252_z": -0.15,
    "sharpe_63_z": 0.10, "vpt_21_z": 0.05, "obv_slope_21_z": 0.05,
}

# ── Walk-forward validation settings ──
WF_VAL_MONTHS = 6          # hold out last 6 months for validation during training
WF_EVAL_MONTHS = 3         # look at last 3 live months to measure recent accuracy
ML_CONFIDENCE_FLOOR = 0.30 # never trust ML less than 30% (always blend some ML in)
ML_CONFIDENCE_CAP = 0.85   # never trust ML more than 85% (always blend some fallback)

# --- extra risk caps (new) ---
MAX_STOCK_WEIGHT = 0.12      # cap single-name stocks
MAX_ETF_WEIGHT   = 0.18      # allow bigger ETFs (SHY/IEF/QQQ etc)
MAX_SLV_WEIGHT   = 0.10      # cap silver ETF
SEMIS_MAX_NAMES  = 2         # max 2 semi/storage names
SEMIS_MAX_TOTAL  = 0.25      # max 25% total weight in semi/storage bucket

SEMIS_STORAGE = {
    "mu.us","wdc.us","stx.us","asml.us","intc.us","nvda.us","amd.us",
    "lrcx.us","klac.us","amat.us","txn.us","qcom.us","avgo.us","mrvl.us"
}

KNOWN_ETFS = {
    "spy.us","qqq.us","dia.us","iwm.us","vti.us","voo.us",
    "ief.us","tlt.us","shy.us","lqd.us","hyg.us",
    "gld.us","iau.us","slv.us",
    "dbc.us","pdbc.us","djp.us","uso.us","ung.us","dba.us","cper.us","copx.us",
}

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _group_of(t: str) -> str:
    return GROUP_MAP.get(t, DEFAULT_GROUP)

def _bucket_of(t: str) -> str:
    g = _group_of(t)
    if g == "bonds": return "bonds"
    if g in {"gold", "metals", "commodities", "energy", "agriculture"}: return "commodities"
    return "equity"

def _normalize(w: pd.Series) -> pd.Series:
    s = float(w.sum())
    if s <= 0 or np.isnan(s): return w * 0.0
    return w / s

def _zscore_by_date(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns: continue
        out[c + "_z"] = out.groupby("date")[c].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12)
        )
    return out


# ═══════════════════════════════════════════════════════════
# Layer 1: ML scoring — walk-forward validated ensemble
# ═══════════════════════════════════════════════════════════

def _prepare_ml_target(month_end: pd.DataFrame) -> pd.DataFrame:
    df = month_end.copy()
    df = df.sort_values(["ticker", "date"])
    df["fwd_ret_1m"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1.0
    return df


def _train_lgbm_with_validation(train_df: pd.DataFrame, val_df: pd.DataFrame,
                                 n_estimators: int = 300) -> Tuple[object, float]:
    """
    Train LightGBM with early stopping on a held-out validation set.
    Returns (model, val_rank_corr) where val_rank_corr measures how well
    the model's predictions correlate with actual forward returns on unseen data.
    A higher rank correlation = better model.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None, 0.0

    avail = [f for f in ML_FEATURES if f in train_df.columns]
    if len(avail) < 5:
        return None, 0.0

    # Clean training data
    tr = train_df.dropna(subset=avail + ["fwd_ret_1m"]).copy()
    if len(tr) < ML_MIN_TRAIN_SAMPLES:
        return None, 0.0

    X_train = tr[avail].values
    y_train = tr["fwd_ret_1m"].rank(pct=True).values

    # Clean validation data
    vl = val_df.dropna(subset=avail + ["fwd_ret_1m"]).copy()
    if len(vl) < 20:
        # Not enough validation data — train without early stopping
        model = lgb.LGBMRegressor(
            n_estimators=min(n_estimators, 150),
            max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=1,
        )
        model.fit(X_train, y_train)
        model._feature_names = avail
        return model, 0.5  # neutral confidence

    X_val = vl[avail].values
    y_val = vl["fwd_ret_1m"].rank(pct=True).values

    # Train with early stopping — stops adding trees when validation error stops improving
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.7, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    model._feature_names = avail

    # Measure validation accuracy: rank correlation between predicted and actual
    preds = model.predict(X_val)
    try:
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(preds, vl["fwd_ret_1m"].values)
        if np.isnan(rank_corr):
            rank_corr = 0.0
    except ImportError:
        # No scipy — approximate with pandas rank correlation
        rank_corr = float(pd.Series(preds).corr(vl["fwd_ret_1m"].reset_index(drop=True), method="spearman"))
        if np.isnan(rank_corr):
            rank_corr = 0.0

    return model, float(rank_corr)


def _predict_scores(model, snap: pd.DataFrame) -> pd.Series:
    avail = model._feature_names
    X = snap[avail].fillna(0.0).values
    scores = model.predict(X)
    return pd.Series(scores, index=snap.index, name="ml_score")


def _train_validated_ensemble(month_end: pd.DataFrame, current_month,
                               windows: List[int]) -> Tuple[List[object], float]:
    """
    Train multiple LightGBM models on different lookback windows,
    each with a proper train/validation split.

    Returns (models, avg_confidence) where avg_confidence is the average
    validation rank correlation across all successfully trained models.
    This tells us how much to trust the ML signal.
    """
    models = []
    confidences = []

    for w in windows:
        train_start = current_month - w
        full_window = month_end[
            (month_end["month"] >= train_start) &
            (month_end["month"] < current_month)
        ].copy()

        if len(full_window) < ML_MIN_TRAIN_SAMPLES:
            models.append(None)
            continue

        # Split: everything except last WF_VAL_MONTHS for training,
        # last WF_VAL_MONTHS for validation (purged — no overlap)
        val_start = current_month - WF_VAL_MONTHS
        train_df = full_window[full_window["month"] < val_start].copy()
        val_df = full_window[full_window["month"] >= val_start].copy()

        if len(train_df) < ML_MIN_TRAIN_SAMPLES // 2:
            # Not enough training data after split — use full window without validation
            train_df = full_window
            val_df = pd.DataFrame()

        n_est = min(300, max(80, w * 5))
        m, rc = _train_lgbm_with_validation(train_df, val_df, n_estimators=n_est)
        models.append(m)
        if m is not None:
            confidences.append(rc)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return models, avg_conf


def _ensemble_predict(models: List[object], snap: pd.DataFrame) -> pd.Series:
    preds = []
    for m in models:
        if m is None: continue
        try:
            p = _predict_scores(m, snap)
            preds.append(p)
        except Exception:
            continue
    if not preds: return None
    return pd.concat(preds, axis=1).mean(axis=1)


def _compute_ml_confidence(val_confidence: float, recent_accuracy: float) -> float:
    """
    Combine validation-set confidence with recent live accuracy
    to decide how much to trust the ML vs fallback.

    val_confidence: rank correlation on held-out validation set (0 to ~0.3 typical)
    recent_accuracy: how well recent live predictions matched reality (0 to 1)

    Returns a blend weight between ML_CONFIDENCE_FLOOR and ML_CONFIDENCE_CAP.
    """
    # Validation rank corr of 0.10+ is decent for financial data
    # Map 0.0 → 0.0, 0.05 → 0.3, 0.10 → 0.6, 0.15+ → 1.0
    val_score = float(np.clip((val_confidence - 0.0) / 0.15, 0.0, 1.0))

    # Blend validation confidence with recent accuracy (if available)
    if recent_accuracy > 0:
        combined = 0.6 * val_score + 0.4 * recent_accuracy
    else:
        combined = val_score

    # Map to confidence range
    confidence = ML_CONFIDENCE_FLOOR + combined * (ML_CONFIDENCE_CAP - ML_CONFIDENCE_FLOOR)
    return float(np.clip(confidence, ML_CONFIDENCE_FLOOR, ML_CONFIDENCE_CAP))


def _measure_recent_accuracy(month_end: pd.DataFrame, models: List[object],
                              current_month, eval_months: int = 3) -> float:
    """
    Look at the last eval_months of actual results and measure how well
    the ensemble's predictions matched reality. Returns 0-1 score.

    This is the "did our recent picks actually work?" check.
    """
    if not models or not any(m is not None for m in models):
        return 0.0

    recent_months = []
    for offset in range(1, eval_months + 1):
        m = current_month - offset
        snap = month_end[month_end["month"] == m].copy()
        if snap.empty or "fwd_ret_1m" not in snap.columns:
            continue
        snap = snap.dropna(subset=["fwd_ret_1m"])
        if len(snap) < 10:
            continue
        recent_months.append(snap)

    if not recent_months:
        return 0.0

    # For each recent month, predict scores and see if top-ranked actually outperformed
    hit_rates = []
    for snap in recent_months:
        pred = _ensemble_predict(models, snap)
        if pred is None or len(pred) < 10:
            continue

        snap = snap.copy()
        snap["pred_rank"] = pred.rank(ascending=False)
        snap["actual_rank"] = snap["fwd_ret_1m"].rank(ascending=False)

        # Did the top quintile (top 20%) of predictions actually outperform the bottom?
        n = len(snap)
        top_cutoff = max(1, n // 5)
        top_pred = snap.nsmallest(top_cutoff, "pred_rank")
        bottom_pred = snap.nlargest(top_cutoff, "pred_rank")

        top_actual = top_pred["fwd_ret_1m"].mean()
        bottom_actual = bottom_pred["fwd_ret_1m"].mean()

        # If top predictions beat bottom predictions, that's a hit
        if top_actual > bottom_actual:
            # Scale by how much — bigger spread = more confident
            spread = top_actual - bottom_actual
            hit_rates.append(min(1.0, 0.5 + spread * 10))
        else:
            hit_rates.append(max(0.0, 0.5 + (top_actual - bottom_actual) * 10))

    if not hit_rates:
        return 0.0

    return float(np.mean(hit_rates))


def _fallback_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for col, wt in FALLBACK_WEIGHTS.items():
        if col in df.columns:
            score = score + wt * df[col].astype(float).fillna(0.0)
    return score


# ═══════════════════════════════════════════════════════════
# News sentiment overlay
# ═══════════════════════════════════════════════════════════

NEWS_OVERLAY_WEIGHT = 0.10

def _apply_news_overlay(snap: pd.DataFrame) -> pd.DataFrame:
    out = snap.copy()
    if "sent_mean_7d" not in out.columns:
        return out

    for col in ["sent_mean_7d", "sent_mean_30d", "sent_shock", "news_count_7d"]:
        if col in out.columns:
            vals = out[col].astype(float)
            mu, std = vals.mean(), vals.std()
            out[col + "_z"] = (vals - mu) / (std + 1e-12) if std > 1e-12 else 0.0

    news_score = pd.Series(0.0, index=out.index)
    if "sent_mean_7d_z" in out.columns:
        news_score += 0.4 * out["sent_mean_7d_z"].fillna(0.0)
    if "sent_shock_z" in out.columns:
        news_score += 0.4 * out["sent_shock_z"].fillna(0.0)
    if "news_count_7d_z" in out.columns:
        news_score += 0.2 * out["news_count_7d_z"].fillna(0.0)

    if "score" in out.columns:
        out["score"] = (1.0 - NEWS_OVERLAY_WEIGHT) * out["score"] + NEWS_OVERLAY_WEIGHT * news_score
    return out


# ═══════════════════════════════════════════════════════════
# Layer 2: Volatility targeting
# ═══════════════════════════════════════════════════════════

def _vol_scale_factor(snap: pd.DataFrame) -> float:
    spy = snap[snap["ticker"] == "spy.us"]
    if spy.empty: return 1.0
    vol_col = "vol_63" if "vol_63" in spy.columns else None
    if vol_col is None: return 1.0
    rv = float(spy[vol_col].iloc[0])
    if np.isnan(rv) or rv <= 0: return 1.0
    ann_vol = rv * np.sqrt(252)
    return float(np.clip(TARGET_VOL / ann_vol, VOL_FLOOR, VOL_CAP))


# ═══════════════════════════════════════════════════════════
# Layer 3: Drawdown circuit breaker
# ═══════════════════════════════════════════════════════════

def _drawdown_tilt(snap: pd.DataFrame) -> float:
    spy = snap[snap["ticker"] == "spy.us"]
    if spy.empty: return 0.0
    dd_col = "maxdd_63" if "maxdd_63" in spy.columns else "maxdd_252"
    if dd_col not in spy.columns: return 0.0
    dd = float(spy[dd_col].iloc[0])
    if np.isnan(dd): return 0.0
    if dd >= DD_TILT_START: return 0.0
    if dd <= DD_TILT_MAX: return DD_MAX_SHIFT
    frac = (DD_TILT_START - dd) / (DD_TILT_START - DD_TILT_MAX)
    return float(frac * DD_MAX_SHIFT)

def _adjusted_budgets(snap: pd.DataFrame) -> Dict[str, float]:
    b = dict(BUDGETS_BASE)
    vs = _vol_scale_factor(snap)
    b["equity"] *= vs
    tilt = _drawdown_tilt(snap)
    shift = min(tilt, b["equity"] - EQUITY_FLOOR)
    shift = max(shift, 0.0)
    b["equity"] -= shift
    b["bonds"] += shift
    total = sum(b.values())
    if total > 0: b = {k: v / total for k, v in b.items()}
    return b


# ═══════════════════════════════════════════════════════════
# Correlation filter + sector caps
# ═══════════════════════════════════════════════════════════

def _trailing_returns_pivot(prices_df: pd.DataFrame, end_date: pd.Timestamp, window: int) -> pd.DataFrame:
    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= end_date].sort_values(["date", "ticker"])
    if df.empty: return pd.DataFrame()
    start_cut = end_date - pd.Timedelta(days=int(window * 1.8))
    df = df[df["date"] >= start_cut]
    px = df.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = px.pct_change().dropna(how="all")
    if len(rets) > window: rets = rets.iloc[-window:]
    return rets

def _corr_filter_select_with_sector_cap(
    candidates: pd.DataFrame, rets: pd.DataFrame,
    k: int, threshold: float, max_per_sector: int,
) -> List[str]:
    if candidates.empty: return []
    ordered = candidates.sort_values("score", ascending=False).copy()
    ordered["sector"] = ordered["ticker"].map(get_sector)

    if rets.empty:
        chosen, sector_count = [], {}
        for _, row in ordered.iterrows():
            if len(chosen) >= k: break
            sec = row["sector"]
            if sector_count.get(sec, 0) >= max_per_sector: continue
            chosen.append(row["ticker"])
            sector_count[sec] = sector_count.get(sec, 0) + 1
        return chosen

    def max_corr(t, chosen_list):
        if not chosen_list: return 0.0
        cols = [c for c in chosen_list if c in rets.columns]
        if t not in rets.columns or not cols: return 0.0
        c = rets[cols + [t]].corr().get(t, pd.Series(dtype=float)).drop(index=t, errors="ignore")
        if c.empty: return 0.0
        vals = c.values[np.isfinite(c.values)]
        return float(np.max(np.abs(vals))) if len(vals) > 0 else 0.0

    for thresh in [threshold, 0.85, 0.95, 0.99]:
        chosen, sector_count = [], {}
        for _, row in ordered.iterrows():
            if len(chosen) >= k: break
            t, sec = row["ticker"], row["sector"]
            if sector_count.get(sec, 0) >= max_per_sector: continue
            if max_corr(t, chosen) > thresh + 1e-12: continue
            chosen.append(t)
            sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(chosen) >= k: break
    return chosen[:k]


# ═══════════════════════════════════════════════════════════
# Weight construction
# ═══════════════════════════════════════════════════════════

def _weights_with_caps(scores: pd.Series, vols: pd.Series, budget: float) -> pd.Series:
    s = scores.astype(float).copy()
    v = vols.astype(float).copy()
    v = v.replace([np.inf, -np.inf], np.nan).fillna(v.median() if np.isfinite(v.median()) else 0.02)
    v = v.clip(lower=1e-6)
    raw = s.clip(lower=0.0) / v
    if float(raw.sum()) <= 0: raw = 1.0 / v
    w = _normalize(raw) * float(budget)

    for _ in range(20):
        over = w > MAX_WEIGHT
        if not over.any(): break
        excess = float((w[over] - MAX_WEIGHT).sum())
        w[over] = MAX_WEIGHT
        under = w < MAX_WEIGHT
        if excess <= 0 or float(w[under].sum()) <= 0: break
        w[under] += (w[under] / float(w[under].sum())) * excess

    k = len(w)
    min_w = min(MIN_WEIGHT, 0.9 * float(budget) / max(k, 1))
    w = w.clip(lower=min_w)
    w = w / float(w.sum()) * float(budget)

    for _ in range(20):
        over = w > MAX_WEIGHT
        if not over.any(): break
        excess = float((w[over] - MAX_WEIGHT).sum())
        w[over] = MAX_WEIGHT
        under = w < MAX_WEIGHT
        if excess <= 0 or float(w[under].sum()) <= 0: break
        w[under] += (w[under] / float(w[under].sum())) * excess
    return w


# ═══════════════════════════════════════════════════════════
# Hysteresis
# ═══════════════════════════════════════════════════════════

def _apply_hysteresis(candidates: pd.DataFrame, prev_holdings: set, boost: float = 0.15) -> pd.DataFrame:
    out = candidates.copy()
    if "score" not in out.columns or not prev_holdings: return out
    is_held = out["ticker"].isin(prev_holdings)
    out.loc[is_held, "score"] = out.loc[is_held, "score"] + boost
    return out

def _is_etf(t: str) -> bool:
    t = (t or "").lower()
    if t in KNOWN_ETFS:
        return True
    g = _group_of(t)
    return g in {"bonds", "gold", "metals", "commodities", "energy", "agriculture"}

def _cap_redistribute(w: pd.Series, caps: pd.Series) -> pd.Series:
    """
    Cap weights at caps[ticker] and redistribute excess pro-rata to those under cap.
    Assumes w sums to 1.
    """
    w = w.copy().astype(float)
    caps = caps.reindex(w.index).astype(float)

    for _ in range(50):
        over = w > caps + 1e-12
        if not over.any():
            break

        excess = float((w[over] - caps[over]).sum())
        w[over] = caps[over]

        under = w < caps - 1e-12
        room = (caps[under] - w[under]).clip(lower=0.0)
        room_sum = float(room.sum())

        if excess <= 0 or room_sum <= 1e-12:
            break

        w[under] = w[under] + room / room_sum * excess

    # final renorm (numerical safety)
    s = float(w.sum())
    return w / s if s > 0 else w

def _enforce_semis_name_cap(selected: List[str], candidates: pd.DataFrame) -> List[str]:
    """
    Keep at most SEMIS_MAX_NAMES from SEMIS_STORAGE.
    Fill removed slots with next best non-semis candidates.
    """
    out = []
    semis = 0
    for t in selected:
        if t in SEMIS_STORAGE:
            if semis >= SEMIS_MAX_NAMES:
                continue
            semis += 1
        out.append(t)

    need = len(selected) - len(out)
    if need <= 0:
        return out

    # fill from candidates by score, avoiding already selected + avoiding semis overflow
    for t in candidates.sort_values("score", ascending=False)["ticker"].tolist():
        if t in out:
            continue
        if t in SEMIS_STORAGE and semis >= SEMIS_MAX_NAMES:
            continue
        if t in SEMIS_STORAGE:
            semis += 1
        out.append(t)
        if len(out) >= len(selected):
            break

    return out
# ═══════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════

def make_monthly_recommendations(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    V4 pipeline:
    1. Month-end snapshots + z-scores + sector-relative features
    2. Walk-forward validated multi-timeframe LightGBM ensemble
    3. Confidence-weighted blend: ML score vs fallback (based on validation accuracy)
    4. News sentiment overlay
    5. Hysteresis + correlation + sector-capped selection
    6. Vol-targeted, drawdown-adjusted budgets
    7. Weight construction with caps
    """
    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.lower()

    df["month"] = df["date"].dt.to_period("M")
    month_end = df.sort_values("date").groupby(["ticker", "month"]).tail(1).copy()

    # Ensure z-score columns exist
    raw_cols = [
        "mom_12_1", "mom_6_1", "mom_3_1", "mom_1_0",
        "ma_200_ratio", "trend_50_200",
        "vol_63", "vol_252", "maxdd_252", "maxdd_63",
        "sharpe_63", "sharpe_252",
        "mr_zscore_21", "rsi_14",
        "skew_63", "gap_ratio_63", "vol_ratio_21_63",
        "rvol_20", "vpt_21", "obv_slope_21", "vol_vol_21",
        "spy_mom_1m", "spy_vol_63", "yield_curve", "credit_spread", "gold_trend",
    ]
    missing_z = [c for c in raw_cols if (c + "_z") not in month_end.columns and c in month_end.columns]
    if missing_z:
        month_end = _zscore_by_date(month_end, missing_z)

    month_end = _prepare_ml_target(month_end)
    month_end["bucket"] = month_end["ticker"].map(_bucket_of)

    prices_for_corr = df[["date", "ticker", "close"]].copy() if "close" in df.columns else pd.DataFrame()

    all_recs: List[Dict] = []
    all_months = sorted(month_end["month"].unique())
    ensemble_models = []
    ml_confidence = 0.5  # start neutral
    prev_month_holdings = set()

    for i, month in enumerate(all_months):
        snap = month_end[month_end["month"] == month].copy()
        if snap.empty: continue

        end_date = pd.to_datetime(snap["date"].max())
        asof_date = end_date.date().isoformat()

        # ── Layer 1: Walk-forward validated ensemble ──
        new_models, val_confidence = _train_validated_ensemble(
            month_end, month, ML_ENSEMBLE_WINDOWS
        )
        if any(m is not None for m in new_models):
            ensemble_models = new_models

            # Measure recent live accuracy
            recent_acc = _measure_recent_accuracy(month_end, ensemble_models, month, WF_EVAL_MONTHS)

            # Compute confidence: how much to trust ML vs fallback
            ml_confidence = _compute_ml_confidence(val_confidence, recent_acc)

        # Score with ensemble
        ensemble_score = _ensemble_predict(ensemble_models, snap) if ensemble_models else None

        if ensemble_score is not None:
            ml_score = ensemble_score
            fb_score = _fallback_score(snap)

            # Confidence-weighted blend: if ML has been accurate, trust it more
            snap["score"] = ml_confidence * ml_score + (1.0 - ml_confidence) * fb_score
            scoring_method = f"ensemble(conf={ml_confidence:.0%})"
        else:
            snap["score"] = _fallback_score(snap)
            ml_confidence = ML_CONFIDENCE_FLOOR
            scoring_method = "fallback_zscore"

        # ── News overlay ──
        snap = _apply_news_overlay(snap)

        # ── Layer 2+3: Adjusted budgets ──
        budgets = _adjusted_budgets(snap)

        # ── Candidate pools ──
        eq = snap[snap["bucket"] == "equity"].copy()
        bd = snap[snap["bucket"] == "bonds"].sort_values("score", ascending=False)
        cm = snap[snap["bucket"] == "commodities"].sort_values("score", ascending=False)

        # Hysteresis + sector-capped selection
        eq = _apply_hysteresis(eq, prev_month_holdings, boost=0.15)
        eq = eq.sort_values("score", ascending=False).head(80)

        rets = pd.DataFrame()
        if not prices_for_corr.empty:
            rets = _trailing_returns_pivot(prices_for_corr, end_date, CORR_WINDOW_DAYS)

        eq_selected = _corr_filter_select_with_sector_cap(
            eq, rets, EQUITY_K, CORR_THRESHOLD, MAX_PER_SECTOR
        )
        eq_selected = _enforce_semis_name_cap(eq_selected, eq)
        bd_selected = bd["ticker"].tolist()[:BONDS_K]
        cm_selected = cm["ticker"].tolist()[:COMMS_K]

        # ── Weight construction ──
        # ── Weight construction (NEW: optimiser) ──

        selected = list(eq_selected) + list(bd_selected) + list(cm_selected)
        selected = [t.lower() for t in selected]
        selected = list(dict.fromkeys(selected))[:TOTAL_K]

        # Expected returns from the model score
        # Keep it simple: treat score as a return ranking proxy
        mu = snap.set_index("ticker").reindex(selected)["score"].astype(float).fillna(0.0)

        # Daily returns matrix for risk model
        if prices_for_corr.empty:
            rets = pd.DataFrame()
        else:
            rets = _trailing_returns_pivot(prices_for_corr, end_date, CORR_WINDOW_DAYS)

        # Caps per ticker
        caps = pd.Series(
            {t: (MAX_ETF_WEIGHT if _is_etf(t) else MAX_STOCK_WEIGHT) for t in selected},
            index=selected,
            dtype=float,
        )
        if "slv.us" in caps.index:
            caps["slv.us"] = min(caps["slv.us"], MAX_SLV_WEIGHT)

        # Sector caps, convert "max per sector" into a weight cap
        # Simple default: no sector above 35%, you can tune this later
        sector_caps = {}
        for t in selected:
            sec = get_sector(t)
            if sec not in sector_caps:
                sector_caps[sec] = 0.35

        sector_map = {t: get_sector(t) for t in selected}

        # Group caps
        group_caps = {
            "semis_storage": (SEMIS_STORAGE, SEMIS_MAX_TOTAL),
        }

        # Previous weights for turnover penalty
        prev_w = pd.Series(0.0, index=selected)
        if prev_month_holdings:
            # equal weight previous holdings as a proxy, you can store actual prev weights later
            prev_list = [t for t in selected if t in prev_month_holdings]
            if prev_list:
                prev_w.loc[prev_list] = 1.0 / len(prev_list)

        # Min weight, keep your 3% floor
        min_w = float(MIN_WEIGHT)

        res = optimise_long_only(
            tickers=selected,
            mu=mu,
            returns=rets,
            caps=caps,
            min_w=float(MIN_WEIGHT),
            sector_map=sector_map,
            sector_caps=sector_caps,
            group_caps=group_caps,
            prev_w=prev_w,
            risk_aversion=10.0,
            turnover_penalty=0.6,
            l2_penalty=0.05,
            cov_shrink=0.15,
        )

        weights = res.weights.sort_values(ascending=False)
        weights = weights / float(weights.sum())

        # ── Extra risk caps (NEW) ─────────────────────────────
        # 1) cap by instrument type (stock vs ETF) + special cap for SLV
        caps = pd.Series(
            {t: (MAX_ETF_WEIGHT if _is_etf(t) else MAX_STOCK_WEIGHT) for t in weights.index},
            index=weights.index,
            dtype=float,
        )
        if "slv.us" in caps.index:
            caps["slv.us"] = min(caps["slv.us"], MAX_SLV_WEIGHT)

        weights = _cap_redistribute(weights, caps)

        # 2) cap total semi/storage exposure
        semis_names = [t for t in weights.index if t in SEMIS_STORAGE]
        if semis_names:
            semis_total = float(weights.loc[semis_names].sum())
            if semis_total > SEMIS_MAX_TOTAL + 1e-12:
                # shrink semis to the cap
                shrink = SEMIS_MAX_TOTAL / semis_total
                weights.loc[semis_names] *= shrink

                # redistribute removed weight to non-semis (pro-rata), then re-cap
                non = [t for t in weights.index if t not in SEMIS_STORAGE]
                if non:
                    weights.loc[non] *= (1.0 / float(weights.loc[non].sum())) * (1.0 - float(weights.loc[semis_names].sum()))

                weights = _cap_redistribute(weights / float(weights.sum()), caps)

        # 3) final renorm (safety)
        weights = weights / float(weights.sum())
        # ── Build output ──
        vol_scale = _vol_scale_factor(snap)
        dd_shift = _drawdown_tilt(snap)

        for tkr, w in weights.items():
            row = snap[snap["ticker"] == tkr]
            sc = float(row["score"].iloc[0]) if not row.empty else 0.0
            bucket = _bucket_of(tkr)
            parts = []

            parts.append(f"Scoring: {scoring_method}.")
            if bucket == "equity":
                parts.append(f"Selected as top equity holding (corr-filtered, {EQUITY_K} slots).")
            elif bucket == "bonds":
                parts.append("Bond allocation for stability.")
            else:
                parts.append("Commodity allocation for diversification.")

            mom12 = row.get("mom_12_1", pd.Series([np.nan])).iloc[0] if not row.empty else np.nan
            ma200 = row.get("ma_200_ratio", pd.Series([np.nan])).iloc[0] if not row.empty else np.nan
            if pd.notna(mom12): parts.append(f"12m momentum: {float(mom12):.2f}.")
            if pd.notna(ma200): parts.append("Above 200d MA." if float(ma200) > 1.0 else "Below 200d MA.")

            rvol = row.get("rvol_20", pd.Series([np.nan])).iloc[0] if not row.empty else np.nan
            if pd.notna(rvol) and rvol > 1.5: parts.append("Unusually high recent trading volume.")

            spy_vol = row.get("spy_vol_63", pd.Series([np.nan])).iloc[0] if not row.empty else np.nan
            if pd.notna(spy_vol) and spy_vol > 0.20: parts.append("Elevated market volatility environment.")

            parts.append(f"Vol scale: {vol_scale:.2f}x.")
            if dd_shift > 0.001: parts.append(f"DD tilt: shifted {dd_shift*100:.1f}% to bonds.")
            parts.append(f"Target weight: {float(w)*100:.1f}%.")

            eq_pct, bd_pct, cm_pct = budgets['equity']*100, budgets['bonds']*100, budgets['commodities']*100
            parts.append(f"Budgets: eq {eq_pct:.0f}% / bd {bd_pct:.0f}% / cm {cm_pct:.0f}%.")

            all_recs.append({
                "asof_date": asof_date, "ticker": tkr, "action": "BUY_OR_HOLD",
                "score": sc, "target_weight": float(w), "reasons": " ".join(parts),
            })

        prev_month_holdings = set(weights.index.tolist())

    out = pd.DataFrame(all_recs)
    if out.empty: return out
    out["target_weight"] = out["target_weight"].astype(float)
    return out.sort_values(["asof_date", "target_weight"], ascending=[True, False]).reset_index(drop=True)