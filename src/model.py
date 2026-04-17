# src/model.py — V4: walk-forward validated multi-timeframe ensemble (+ cash sleeve max 50%)
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .optimiser import optimise_long_only
from .portfolio_tracking import get_learning_adjustments
from .config import (
    EQUITY_K, BONDS_K, COMMS_K, TOTAL_K,
    MAX_WEIGHT, MIN_WEIGHT,
    CORR_WINDOW_DAYS, CORR_THRESHOLD,
    BUDGETS_BASE,
    TARGET_VOL, VOL_LOOKBACK, VOL_FLOOR, VOL_CAP,
    DD_TILT_START, DD_TILT_MAX, DD_MAX_SHIFT, EQUITY_FLOOR,
    ML_MIN_TRAIN_SAMPLES, ML_ENSEMBLE_WINDOWS,
    CASH_TICKER, CASH_MAX_WEIGHT,
)
from .risk_policy import GROUP_MAP, DEFAULT_GROUP, get_sector, MAX_PER_SECTOR

warnings.filterwarnings("ignore", category=UserWarning)

# ── ML feature columns ──
ML_FEATURES = [
    "mom_12_1_z", "mom_6_1_z", "mom_3_1_z", "mom_1_0_z",
    "ma_200_ratio_z", "trend_50_200_z", "high_52w_ratio_z",
    "vol_63_z", "vol_252_z", "vol_of_vol_21_z", "maxdd_252_z", "maxdd_63_z",
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
    "high_52w_ratio_z": 0.08,
    "vol_63_z": -0.15, "vol_of_vol_21_z": -0.06, "maxdd_252_z": -0.15,
    "sharpe_63_z": 0.10, "vpt_21_z": 0.05, "obv_slope_21_z": 0.05,
}

WF_VAL_MONTHS = 6
WF_EVAL_MONTHS = 3
ML_CONFIDENCE_FLOOR = 0.30
ML_CONFIDENCE_CAP = 0.85

MAX_STOCK_WEIGHT = 0.12
MAX_ETF_WEIGHT = 0.15
MAX_SLV_WEIGHT = 0.08
# Keep performance feedback modest: momentum is already captured by ML features,
# so the overlay should not double-count winners or chase reversals.
LEARNING_OVERLAY_SCALE = 1.00
LEARNING_OVERLAY_CAP = 0.35

SEMIS_MAX_NAMES = 2
SEMIS_MAX_TOTAL = 0.20
SEMIS_STORAGE = {
    "mu.us", "wdc.us", "stx.us", "asml.us", "intc.us", "nvda.us", "amd.us",
    "lrcx.us", "klac.us", "amat.us", "txn.us", "qcom.us", "avgo.us", "mrvl.us",
}

KNOWN_ETFS = {
    "spy.us", "qqq.us", "ief.us", "tlt.us", "shy.us",
    "gld.us", "slv.us", "dbc.us",
    "copx.us",
}


def _group_of(t: str) -> str:
    return GROUP_MAP.get(t, DEFAULT_GROUP)


def _bucket_of(t: str) -> str:
    t = (t or "").lower()
    if t == CASH_TICKER:
        return "cash"
    g = _group_of(t)
    if g == "bonds":
        return "bonds"
    if g in {"gold", "metals", "commodities", "energy", "agriculture"}:
        return "commodities"
    return "equity"


def _normalize(w: pd.Series) -> pd.Series:
    s = float(w.sum())
    if s <= 0 or np.isnan(s):
        return w * 0.0
    return w / s


def _zscore_by_date(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c + "_z"] = out.groupby("date")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def _prepare_ml_target(month_end: pd.DataFrame) -> pd.DataFrame:
    df = month_end.copy().sort_values(["ticker", "date"])
    df["fwd_ret_1m"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1.0
    return df


def _train_lgbm_with_validation(train_df: pd.DataFrame, val_df: pd.DataFrame, n_estimators: int = 300) -> Tuple[object, float]:
    try:
        import lightgbm as lgb
    except ImportError:
        return None, 0.0

    avail = [f for f in ML_FEATURES if f in train_df.columns]
    if len(avail) < 5:
        return None, 0.0

    tr = train_df.dropna(subset=avail + ["fwd_ret_1m"]).copy()
    if len(tr) < ML_MIN_TRAIN_SAMPLES:
        return None, 0.0

    X_train = tr[avail].values
    y_train = tr["fwd_ret_1m"].rank(pct=True).values

    vl = val_df.dropna(subset=avail + ["fwd_ret_1m"]).copy()
    if len(vl) < 20:
        model = lgb.LGBMRegressor(
            n_estimators=min(n_estimators, 150),
            max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=1,
        )
        model.fit(X_train, y_train)
        model._feature_names = avail
        return model, 0.5

    X_val = vl[avail].values
    y_val = vl["fwd_ret_1m"].rank(pct=True).values

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

    preds = model.predict(X_val)
    try:
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(preds, vl["fwd_ret_1m"].values)
        if np.isnan(rank_corr):
            rank_corr = 0.0
    except Exception:
        rank_corr = float(pd.Series(preds).corr(vl["fwd_ret_1m"].reset_index(drop=True), method="spearman"))
        if np.isnan(rank_corr):
            rank_corr = 0.0

    return model, float(rank_corr)


def _train_ridge(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[object, float]:
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        return None, 0.0

    avail = [f for f in ML_FEATURES if f in train_df.columns]
    if len(avail) < 5:
        return None, 0.0

    tr = train_df.dropna(subset=avail + ["fwd_ret_1m"]).copy()
    if len(tr) < ML_MIN_TRAIN_SAMPLES:
        return None, 0.0

    model = Ridge(alpha=1.0)
    X_train = tr[avail].fillna(0.0).values
    y_train = tr["fwd_ret_1m"].rank(pct=True).values

    vl = val_df.dropna(subset=avail + ["fwd_ret_1m"]).copy()
    if len(vl) < 20:
        model.fit(X_train, y_train)
        model._feature_names = avail
        return model, 0.5

    model.fit(X_train, y_train)
    model._feature_names = avail

    preds = model.predict(vl[avail].fillna(0.0).values)
    try:
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(preds, vl["fwd_ret_1m"].values)
        if np.isnan(rank_corr):
            rank_corr = 0.0
    except Exception:
        rank_corr = float(pd.Series(preds).corr(vl["fwd_ret_1m"].reset_index(drop=True), method="spearman"))
        if np.isnan(rank_corr):
            rank_corr = 0.0

    return model, float(rank_corr)


def _predict_scores(model, snap: pd.DataFrame) -> pd.Series:
    avail = model._feature_names
    X = snap[avail].fillna(0.0).values
    scores = model.predict(X)
    return pd.Series(scores, index=snap.index, name="ml_score")


def _train_validated_ensemble(month_end: pd.DataFrame, current_month, windows: List[int]) -> Tuple[List[object], float]:
    models = []
    confidences = []
    max_window = max(windows) if windows else 0
    ridge_window = None

    for w in windows:
        train_start = current_month - w
        full_window = month_end[(month_end["month"] >= train_start) & (month_end["month"] < current_month)].copy()
        full_window = _prepare_ml_target(full_window)
        if w == max_window:
            ridge_window = full_window.copy()

        if len(full_window) < ML_MIN_TRAIN_SAMPLES:
            models.append(None)
            continue

        val_start = current_month - WF_VAL_MONTHS
        no_embargo_train = full_window[full_window["month"] < val_start].copy()
        # One-month embargo avoids validating on feature windows that overlap
        # directly with the latest training month.
        train_df = full_window[full_window["month"] < val_start - 1].copy()
        val_df = full_window[full_window["month"] >= val_start].copy()

        if len(train_df) < ML_MIN_TRAIN_SAMPLES // 2:
            # If the embargo leaves too little history, fall back to the prior
            # no-embargo split rather than discarding validation entirely.
            train_df = no_embargo_train
        if len(train_df) < ML_MIN_TRAIN_SAMPLES // 2:
            train_df = full_window
            val_df = pd.DataFrame()

        n_est = min(300, max(80, w * 5))
        m, rc = _train_lgbm_with_validation(train_df, val_df, n_estimators=n_est)
        models.append(m)
        if m is not None:
            confidences.append(rc)

    if ridge_window is not None and len(ridge_window) >= ML_MIN_TRAIN_SAMPLES:
        val_start = current_month - WF_VAL_MONTHS
        no_embargo_train = ridge_window[ridge_window["month"] < val_start].copy()
        train_df = ridge_window[ridge_window["month"] < val_start - 1].copy()
        val_df = ridge_window[ridge_window["month"] >= val_start].copy()
        if len(train_df) < ML_MIN_TRAIN_SAMPLES // 2:
            train_df = no_embargo_train
        if len(train_df) < ML_MIN_TRAIN_SAMPLES // 2:
            train_df = ridge_window
            val_df = pd.DataFrame()
        m, rc = _train_ridge(train_df, val_df)
        if m is not None:
            models.append(m)
            confidences.append(rc)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return models, avg_conf


def _ensemble_predict(models: List[object], snap: pd.DataFrame) -> pd.Series | None:
    preds = []
    for m in models:
        if m is None:
            continue
        try:
            preds.append(_predict_scores(m, snap))
        except Exception:
            continue
    if not preds:
        return None
    return pd.concat(preds, axis=1).mean(axis=1)


def _measure_recent_accuracy(month_end: pd.DataFrame, models: List[object], current_month, eval_months: int = 3) -> float:
    if not models or not any(m is not None for m in models):
        return 0.0

    eval_start = current_month - eval_months
    eval_window = month_end[(month_end["month"] >= eval_start) & (month_end["month"] <= current_month)].copy()
    eval_window = _prepare_ml_target(eval_window)

    recent_months = []
    for offset in range(1, eval_months + 1):
        m = current_month - offset
        snap = eval_window[eval_window["month"] == m].copy()
        if snap.empty or "fwd_ret_1m" not in snap.columns:
            continue
        snap = snap.dropna(subset=["fwd_ret_1m"])
        if len(snap) < 10:
            continue
        recent_months.append(snap)

    if not recent_months:
        return 0.0

    hit_rates = []
    for snap in recent_months:
        pred = _ensemble_predict(models, snap)
        if pred is None or len(pred) < 10:
            continue

        tmp = snap.copy()
        tmp["pred_rank"] = pred.rank(ascending=False)
        tmp["actual_rank"] = tmp["fwd_ret_1m"].rank(ascending=False)

        n = len(tmp)
        q = max(1, n // 5)
        top_pred = tmp.nsmallest(q, "pred_rank")
        bottom_pred = tmp.nlargest(q, "pred_rank")

        top_actual = top_pred["fwd_ret_1m"].mean()
        bottom_actual = bottom_pred["fwd_ret_1m"].mean()

        spread = top_actual - bottom_actual
        hit_rates.append(float(np.clip(0.5 + spread * 10.0, 0.0, 1.0)))

    return float(np.mean(hit_rates)) if hit_rates else 0.0


def _compute_ml_confidence(val_confidence: float, recent_accuracy: float) -> float:
    val_score = float(np.clip(val_confidence / 0.15, 0.0, 1.0))
    combined = 0.6 * val_score + 0.4 * recent_accuracy if recent_accuracy > 0 else val_score
    confidence = ML_CONFIDENCE_FLOOR + combined * (ML_CONFIDENCE_CAP - ML_CONFIDENCE_FLOOR)
    return float(np.clip(confidence, ML_CONFIDENCE_FLOOR, ML_CONFIDENCE_CAP))


def _fallback_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for col, wt in FALLBACK_WEIGHTS.items():
        if col in df.columns:
            score = score + wt * df[col].astype(float).fillna(0.0)
    return score


NEWS_OVERLAY_WEIGHT = 0.15  # raw ticker sentiment
IMPLICATION_OVERLAY_WEIGHT = 0.18  # event -> cross-asset implication layer


def _apply_news_overlay(snap: pd.DataFrame) -> pd.DataFrame:
    out = snap.copy()
    has_raw_news = "sent_mean_7d" in out.columns
    has_implications = "implication_score_7d" in out.columns or "implication_score_30d" in out.columns
    if not has_raw_news and not has_implications:
        return out

    for col in [
        "sent_mean_7d", "sent_mean_30d", "sent_shock", "news_count_7d",
        "implication_score_7d", "implication_score_30d", "implication_count_7d",
    ]:
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

    base = out["score"].astype(float).fillna(0.0)
    out["score"] = (1.0 - NEWS_OVERLAY_WEIGHT) * base + NEWS_OVERLAY_WEIGHT * news_score

    implication_score = pd.Series(0.0, index=out.index)
    if "implication_score_7d_z" in out.columns:
        implication_score += 0.65 * out["implication_score_7d_z"].fillna(0.0)
    if "implication_score_30d_z" in out.columns:
        implication_score += 0.25 * out["implication_score_30d_z"].fillna(0.0)
    if "implication_count_7d_z" in out.columns:
        implication_score += 0.10 * out["implication_count_7d_z"].fillna(0.0)
    if has_implications:
        # Bound the impact: news can move a candidate up or down materially, but
        # cannot override price/risk controls by itself.
        implication_score = implication_score.clip(-2.0, 2.0)
        out["score"] = (1.0 - IMPLICATION_OVERLAY_WEIGHT) * out["score"] + IMPLICATION_OVERLAY_WEIGHT * implication_score
    return out


def _apply_learning_overlay(snap: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    out = snap.copy()
    learning_map = get_learning_adjustments(asof_date)
    if not learning_map:
        out["learning_adj"] = 0.0
        out["learning_overlay"] = 0.0
        return out
    out["learning_adj"] = out["ticker"].astype(str).str.lower().map(learning_map).fillna(0.0).astype(float)
    base_score = out["score"].astype(float).fillna(0.0)
    rank_pct = base_score.rank(pct=True).fillna(0.5)

    # Only amplify realised winners when the current forward score is clearly
    # strong; otherwise the overlay double-counts momentum and chases reversals.
    positive_gate = np.where(rank_pct >= 0.60, 1.0, 0.25)
    overlay = out["learning_adj"].clip(-LEARNING_OVERLAY_CAP, LEARNING_OVERLAY_CAP) * LEARNING_OVERLAY_SCALE
    overlay = np.where(overlay > 0.0, overlay * positive_gate, overlay)
    out["learning_overlay"] = pd.Series(overlay, index=out.index).clip(-LEARNING_OVERLAY_CAP, LEARNING_OVERLAY_CAP)
    out["score"] = base_score + out["learning_overlay"]
    return out


def _spy_row(snap: pd.DataFrame) -> pd.Series | None:
    spy = snap[snap["ticker"] == "spy.us"]
    if spy.empty:
        return None
    return spy.iloc[0]


def _vol_scale_factor(snap: pd.DataFrame) -> float:
    """
    More responsive vol targeting:
    use max(21d realised vol, 63d realised vol) if available.
    """
    r = _spy_row(snap)
    if r is None:
        return 1.0

    vol21 = float(r.get("vol_21", np.nan))
    vol63 = float(r.get("vol_63", np.nan))

    candidates = []
    if np.isfinite(vol21) and vol21 > 0:
        candidates.append(vol21)
    if np.isfinite(vol63) and vol63 > 0:
        candidates.append(vol63)
    if not candidates:
        return 1.0

    rv = float(max(candidates))
    ann_vol = rv * np.sqrt(252)
    if not np.isfinite(ann_vol) or ann_vol <= 0:
        return 1.0

    return float(np.clip(TARGET_VOL / ann_vol, VOL_FLOOR, VOL_CAP))


def _drawdown_tilt(snap: pd.DataFrame) -> float:
    r = _spy_row(snap)
    if r is None:
        return 0.0

    dd = float(r.get("maxdd_63", r.get("maxdd_252", 0.0)))
    if not np.isfinite(dd):
        return 0.0

    if dd >= DD_TILT_START:
        return 0.0
    if dd <= DD_TILT_MAX:
        return DD_MAX_SHIFT

    frac = (DD_TILT_START - dd) / (DD_TILT_START - DD_TILT_MAX)
    return float(frac * DD_MAX_SHIFT)


def _market_news_stress(snap: pd.DataFrame) -> float:
    """
    0..1 stress from live news:
    uses absolute SPY sentiment shock if present, else cross-sectional average.
    """
    if "sent_shock" not in snap.columns:
        return 0.0

    spy = snap[snap["ticker"] == "spy.us"]
    if not spy.empty:
        s = float(spy["sent_shock"].iloc[0])
        if np.isfinite(s):
            return float(np.clip(abs(s) / 3.0, 0.0, 1.0))

    vals = snap["sent_shock"].astype(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0
    return float(np.clip(vals.abs().mean() / 3.0, 0.0, 1.0))


def _market_implication_stress(snap: pd.DataFrame) -> float:
    if "implication_score_7d" not in snap.columns:
        return 0.0
    vals = snap["implication_score_7d"].astype(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0
    # Negative implication events across the candidate set increase cash stress.
    neg = vals[vals < 0.0]
    if len(neg) == 0:
        return 0.0
    return float(np.clip(abs(float(neg.mean())) / 3.0, 0.0, 1.0))


def _cash_weight(snap: pd.DataFrame) -> float:
    """
    Cash 0..CASH_MAX_WEIGHT driven by:
    - below 200d MA (binary)
    - drawdown stress
    - realised volatility stress
    - live news stress (NEW)
    """
    r = _spy_row(snap)
    if r is None:
        return 0.0

    ma200 = float(r.get("ma_200_ratio", 1.0))
    dd = float(r.get("maxdd_63", r.get("maxdd_252", 0.0)))
    vol63 = float(r.get("vol_63", np.nan))
    vol21 = float(r.get("vol_21", np.nan))

    ma_stress = 1.0 if np.isfinite(ma200) and ma200 < 1.0 else 0.0

    if np.isfinite(dd):
        dd_stress = float(np.clip((DD_TILT_START - dd) / (DD_TILT_START - DD_TILT_MAX + 1e-12), 0.0, 1.0))
    else:
        dd_stress = 0.0

    rv = None
    if np.isfinite(vol21) and vol21 > 0:
        rv = vol21
    if np.isfinite(vol63) and vol63 > 0:
        rv = max(rv, vol63) if rv is not None else vol63

    if rv is None:
        vol_stress = 0.0
    else:
        ann_vol = float(rv * np.sqrt(252))
        vol_stress = float(np.clip((ann_vol - TARGET_VOL) / (TARGET_VOL + 1e-12), 0.0, 1.0))

    news_stress = _market_news_stress(snap)
    implication_stress = _market_implication_stress(snap)

    stress = 0.32 * ma_stress + 0.28 * dd_stress + 0.20 * vol_stress + 0.10 * news_stress + 0.10 * implication_stress
    cash_w = float(np.clip(stress * CASH_MAX_WEIGHT, 0.0, CASH_MAX_WEIGHT))

    # Hard circuit breakers override the soft stress model in fast drawdown environments.
    if np.isfinite(dd):
        if dd < -0.25:
            cash_w = max(cash_w, 0.40)
        elif dd < -0.15:
            cash_w = max(cash_w, 0.25)

    # avoid tiny cash dust
    if cash_w < 0.02:
        cash_w = 0.0
    return cash_w


def _detect_regime(snap: pd.DataFrame) -> str:
    r = _spy_row(snap)
    if r is None:
        return "unknown"

    ma200 = float(r.get("ma_200_ratio", np.nan))
    vol63 = float(r.get("vol_63", np.nan))
    maxdd = float(r.get("maxdd_63", np.nan))
    gold_trend = float(r.get("gold_trend", np.nan))
    yield_curve = float(r.get("yield_curve", np.nan))

    ann_vol = vol63 * np.sqrt(252) if np.isfinite(vol63) and vol63 > 0 else np.nan
    below_ma = np.isfinite(ma200) and ma200 < 1.0
    high_vol = np.isfinite(ann_vol) and ann_vol > 0.22
    low_vol = np.isfinite(ann_vol) and ann_vol < 0.18
    deep_dd = np.isfinite(maxdd) and maxdd < -0.10
    inflation_gold = np.isfinite(gold_trend) and (gold_trend > 1.02 or gold_trend > 0.02)
    inverted_curve = np.isfinite(yield_curve) and yield_curve < 0.0

    if below_ma and high_vol:
        return "bear"
    if below_ma or deep_dd:
        return "risk_off"
    if inflation_gold and inverted_curve:
        return "inflation"
    if np.isfinite(ma200) and ma200 >= 1.0 and low_vol:
        return "bull"
    return "unknown"


def _adjusted_budgets(snap: pd.DataFrame) -> Dict[str, float]:
    b = dict(BUDGETS_BASE)
    regime = _detect_regime(snap)

    if regime == "bear":
        b["equity"] *= 0.70
        b["bonds"] *= 1.30
    elif regime == "risk_off":
        b["equity"] *= 0.80
        b["bonds"] *= 1.20
    elif regime == "inflation":
        b["commodities"] *= 1.40
        b["bonds"] *= 0.80

    cash_w = _cash_weight(snap)
    risky_total = 1.0 - cash_w

    vs = _vol_scale_factor(snap)
    b["equity"] *= vs

    tilt = _drawdown_tilt(snap)
    # enforce equity floor inside risky sleeve
    equity_floor_abs = float(EQUITY_FLOOR) * risky_total
    if risky_total > 1e-12:
        shift = max(0.0, min(tilt, b["equity"] - (equity_floor_abs / risky_total)))
    else:
        shift = 0.0

    b["equity"] -= shift
    b["bonds"] += shift

    risky_sum = float(sum(b.values()))
    if risky_sum > 0:
        b = {k: (v / risky_sum) * risky_total for k, v in b.items()}

    b["cash"] = cash_w

    total = float(sum(b.values()))
    if total > 0:
        b = {k: v / total for k, v in b.items()}
    return b


def _trailing_returns_pivot(prices_df: pd.DataFrame, end_date: pd.Timestamp, window: int) -> pd.DataFrame:
    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= end_date].sort_values(["date", "ticker"])
    if df.empty:
        return pd.DataFrame()

    start_cut = end_date - pd.Timedelta(days=int(window * 1.8))
    df = df[df["date"] >= start_cut]

    px = df.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = px.pct_change(fill_method=None).dropna(how="all")
    if len(rets) > window:
        rets = rets.iloc[-window:]
    return rets


def _corr_filter_select_with_sector_cap(candidates: pd.DataFrame, rets: pd.DataFrame, k: int, threshold: float, max_per_sector: int) -> List[str]:
    if candidates.empty:
        return []

    ordered = candidates.sort_values("score", ascending=False).copy()
    ordered["sector"] = ordered["ticker"].map(get_sector)

    if rets.empty:
        chosen, sector_count = [], {}
        for _, row in ordered.iterrows():
            if len(chosen) >= k:
                break
            sec = row["sector"]
            if sector_count.get(sec, 0) >= max_per_sector:
                continue
            chosen.append(row["ticker"])
            sector_count[sec] = sector_count.get(sec, 0) + 1
        return chosen

    def max_corr(t: str, chosen_list: List[str]) -> float:
        if not chosen_list:
            return 0.0
        cols = [c for c in chosen_list if c in rets.columns]
        if t not in rets.columns or not cols:
            return 0.0
        c = rets[cols + [t]].corr().get(t, pd.Series(dtype=float)).drop(index=t, errors="ignore")
        if c.empty:
            return 0.0
        vals = c.values[np.isfinite(c.values)]
        return float(np.max(np.abs(vals))) if len(vals) > 0 else 0.0

    for thresh in [threshold, 0.85, 0.95, 0.99]:
        chosen, sector_count = [], {}
        for _, row in ordered.iterrows():
            if len(chosen) >= k:
                break
            t, sec = row["ticker"], row["sector"]
            if sector_count.get(sec, 0) >= max_per_sector:
                continue
            if max_corr(t, chosen) > thresh + 1e-12:
                continue
            chosen.append(t)
            sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(chosen) >= k:
            break

    return chosen[:k]


def _apply_hysteresis(candidates: pd.DataFrame, prev_holdings: set, boost: float = 0.15) -> pd.DataFrame:
    out = candidates.copy()
    if "score" not in out.columns or not prev_holdings:
        return out
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

    s = float(w.sum())
    return w / s if s > 0 else w


def _enforce_semis_name_cap(selected: List[str], candidates: pd.DataFrame) -> List[str]:
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


def make_monthly_recommendations(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.lower()

    df["month"] = df["date"].dt.to_period("M")
    month_end = df.sort_values("date").groupby(["ticker", "month"]).tail(1).copy()

    raw_cols = [
        "mom_12_1", "mom_6_1", "mom_3_1", "mom_1_0",
        "ma_200_ratio", "trend_50_200", "high_52w_ratio",
        "vol_21", "vol_63", "vol_252", "vol_of_vol_21", "maxdd_252", "maxdd_63",
        "sharpe_63", "sharpe_252",
        "mr_zscore_21", "rsi_14",
        "skew_63", "gap_ratio_63", "vol_ratio_21_63",
        "rvol_20", "vpt_21", "obv_slope_21", "vol_vol_21",
        "spy_mom_1m", "spy_vol_63", "yield_curve", "credit_spread", "gold_trend",
    ]
    missing_z = [c for c in raw_cols if (c + "_z") not in month_end.columns and c in month_end.columns]
    if missing_z:
        month_end = _zscore_by_date(month_end, missing_z)

    month_end["bucket"] = month_end["ticker"].map(_bucket_of)

    prices_for_corr = df[["date", "ticker", "close"]].copy() if "close" in df.columns else pd.DataFrame()

    all_recs: List[Dict] = []
    all_months = sorted(month_end["month"].unique())
    latest_month = max(all_months) if all_months else None
    ensemble_models: List[object] = []
    ml_confidence = 0.5
    prev_month_holdings: set[str] = set()
    prev_weights_series = pd.Series(dtype=float)

    for month in all_months:
        snap = month_end[month_end["month"] == month].copy()
        if snap.empty:
            continue

        end_date = pd.to_datetime(snap["date"].max())
        # The live recommendation snapshot is dated to the latest close used to build it.
        asof_date = end_date.date().isoformat()

        # Train ensemble
        new_models, val_confidence = _train_validated_ensemble(month_end, month, ML_ENSEMBLE_WINDOWS)
        if any(m is not None for m in new_models):
            ensemble_models = new_models
            recent_acc = _measure_recent_accuracy(month_end, ensemble_models, month, WF_EVAL_MONTHS)
            ml_confidence = _compute_ml_confidence(val_confidence, recent_acc)

        pred = _ensemble_predict(ensemble_models, snap) if ensemble_models else None
        if pred is not None:
            fb = _fallback_score(snap)
            snap["ml_signal"] = pred
            snap["score"] = ml_confidence * pred + (1.0 - ml_confidence) * fb
            scoring_method = f"ensemble(conf={ml_confidence:.0%})"
        else:
            snap["ml_signal"] = np.nan
            snap["score"] = _fallback_score(snap)
            scoring_method = "fallback_zscore"

        # News overlay
        pre_news_score = snap["score"].astype(float).copy()
        snap = _apply_news_overlay(snap)
        snap["news_contrib"] = snap["score"].astype(float) - pre_news_score
        if latest_month is not None and month == latest_month:
            snap = _apply_learning_overlay(snap, asof_date)
        else:
            snap["learning_adj"] = 0.0
            snap["learning_overlay"] = 0.0

        # Budgets (includes cash)
        budgets = _adjusted_budgets(snap)
        regime = _detect_regime(snap)
        cash_w = float(np.clip(budgets.get("cash", 0.0), 0.0, CASH_MAX_WEIGHT))
        risky_total = 1.0 - cash_w

        # Candidate pools
        eq = snap[snap["bucket"] == "equity"].copy()
        bd = snap[snap["bucket"] == "bonds"].sort_values("score", ascending=False)
        cm = snap[snap["bucket"] == "commodities"].sort_values("score", ascending=False)

        eq = _apply_hysteresis(eq, prev_month_holdings, boost=0.03)
        eq = eq.sort_values("score", ascending=False).head(80)

        rets = pd.DataFrame()
        if not prices_for_corr.empty:
            rets = _trailing_returns_pivot(prices_for_corr, end_date, CORR_WINDOW_DAYS)

        eq_selected = _corr_filter_select_with_sector_cap(eq, rets, EQUITY_K, CORR_THRESHOLD, MAX_PER_SECTOR)
        eq_selected = _enforce_semis_name_cap(eq_selected, eq)
        bd_selected = bd["ticker"].tolist()[:BONDS_K]
        cm_selected = cm["ticker"].tolist()[:COMMS_K]

        selected = list(dict.fromkeys([*(eq_selected), *(bd_selected), *(cm_selected)]))[:TOTAL_K]

        # Expected returns proxy
        raw_mu = snap.set_index("ticker").reindex(selected)["score"].astype(float).fillna(0.0)
        # Model scores are dimensionless ranks/z-scores; scale them into a
        # modest expected-return proxy so the MVO risk term remains calibrated
        # against daily-return covariance. A score of 1.0 maps to roughly 4%.
        mu = raw_mu * 0.04

        # Risk model matrix
        rets = _trailing_returns_pivot(prices_for_corr, end_date, CORR_WINDOW_DAYS) if not prices_for_corr.empty else pd.DataFrame()

        # Caps per ticker
        caps = pd.Series({t: (MAX_ETF_WEIGHT if _is_etf(t) else MAX_STOCK_WEIGHT) for t in selected}, index=selected, dtype=float)
        if "slv.us" in caps.index:
            caps["slv.us"] = min(caps["slv.us"], MAX_SLV_WEIGHT)

        # Sector caps
        sector_map = {t: get_sector(t) for t in selected}
        sector_caps = {sec: 0.35 for sec in set(sector_map.values())}
        sector_caps["Other"] = 0.24

        # Group caps
        group_caps = {"semis_storage": (SEMIS_STORAGE, SEMIS_MAX_TOTAL)}

        # Turnover proxy based on actual prior risky-sleeve weights, not
        # equal-weight membership. This makes the optimiser's turnover penalty
        # reflect the previous target portfolio shape.
        prev_w = prev_weights_series.reindex(selected).fillna(0.0).astype(float)
        prev_sum = float(prev_w.sum())
        if prev_sum > 0:
            prev_w = prev_w / prev_sum

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
            risk_aversion=11.0,
            turnover_penalty=0.45,
            l2_penalty=0.05,
            cov_shrink=0.25,
        )

        weights = res.weights
        if weights is None or len(weights) == 0:
            continue

        # Normalise and drop dust
        weights = weights / float(weights.sum())
        weights = weights[weights >= MIN_WEIGHT - 1e-6]
        if weights.empty:
            continue
        weights = weights / float(weights.sum())

        # Cap instrument weights and redistribute
        caps2 = pd.Series({t: (MAX_ETF_WEIGHT if _is_etf(t) else MAX_STOCK_WEIGHT) for t in weights.index}, index=weights.index, dtype=float)
        if "slv.us" in caps2.index:
            caps2["slv.us"] = min(caps2["slv.us"], MAX_SLV_WEIGHT)
        weights = _cap_redistribute(weights, caps2)

        # Semi/storage total cap
        semis_names = [t for t in weights.index if t in SEMIS_STORAGE]
        if semis_names:
            semis_total = float(weights.loc[semis_names].sum())
            if semis_total > SEMIS_MAX_TOTAL + 1e-12:
                shrink = SEMIS_MAX_TOTAL / semis_total
                weights.loc[semis_names] *= shrink
                non = [t for t in weights.index if t not in SEMIS_STORAGE]
                if non:
                    weights.loc[non] *= (1.0 / float(weights.loc[non].sum())) * (1.0 - float(weights.loc[semis_names].sum()))
                weights = _cap_redistribute(weights / float(weights.sum()), caps2)

        weights = weights / float(weights.sum())
        risky_weights = weights.copy()

        # Scale risky sleeve
        weights = weights * risky_total

        # Output rows
        vol_scale = _vol_scale_factor(snap)
        dd_shift = _drawdown_tilt(snap)

        for tkr, w in weights.items():
            row = snap[snap["ticker"] == tkr]
            sc = float(row["score"].iloc[0]) if not row.empty else 0.0
            bucket = _bucket_of(tkr)
            ml_signal = float(row["ml_signal"].iloc[0]) if not row.empty and "ml_signal" in row.columns and pd.notna(row["ml_signal"].iloc[0]) else None
            news_contrib = float(row["news_contrib"].iloc[0]) if not row.empty and "news_contrib" in row.columns else 0.0
            learning_overlay = float(row["learning_overlay"].iloc[0]) if not row.empty and "learning_overlay" in row.columns else 0.0
            pie_w = float(w) / risky_total if risky_total > 1e-12 else 0.0

            parts = [f"Scoring: {scoring_method}."]
            if ml_signal is not None:
                parts.append(f"ML signal: {ml_signal:.2f}.")
            parts.append(f"News adj: {news_contrib:+.2f}.")
            parts.append(f"Learning adj: {learning_overlay:+.3f}.")
            parts.append(f"Regime: {regime}.")
            if bucket == "equity":
                parts.append(f"Selected as top equity holding (corr-filtered, {EQUITY_K} slots).")
            elif bucket == "bonds":
                parts.append("Bond allocation for stability.")
            else:
                parts.append("Commodity allocation for diversification.")

            parts.append(f"Vol scale: {vol_scale:.2f}x.")
            if dd_shift > 0.001:
                parts.append(f"DD tilt: shifted {dd_shift*100:.1f}% to bonds.")
            parts.append(f"Cash sleeve: {cash_w*100:.1f}%.")
            parts.append(f"Account weight: {float(w)*100:.1f}%.")
            parts.append(f"Pie weight: {pie_w*100:.1f}% (cash kept outside pie).")

            all_recs.append(
                {
                    "asof_date": asof_date,
                    "ticker": tkr,
                    "action": "BUY_OR_HOLD",
                    "score": sc,
                    "target_weight": float(w),
                    "reasons": " ".join(parts),
                }
            )

        # Explicit cash row for UI
        if cash_w > 1e-6:
            all_recs.append(
                {
                    "asof_date": asof_date,
                    "ticker": CASH_TICKER,
                    "action": "HOLD_CASH",
                    "score": 0.0,
                    "target_weight": float(cash_w),
                    "reasons": f"Cash sleeve for risk control (capped at {CASH_MAX_WEIGHT*100:.0f}%). Target weight: {cash_w*100:.1f}%.",
                }
            )

        prev_month_holdings = set(weights.index.tolist())
        prev_weights_series = risky_weights.copy()

    out = pd.DataFrame(all_recs)
    if out.empty:
        return out
    out["target_weight"] = out["target_weight"].astype(float)
    sums = out.groupby("asof_date")["target_weight"].transform("sum")
    out["target_weight"] = np.where(sums > 0, out["target_weight"] / sums, out["target_weight"])
    return out.sort_values(["asof_date", "target_weight"], ascending=[True, False]).reset_index(drop=True)
