# src/features.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_policy import GROUP_MAP, DEFAULT_GROUP


SECTOR_MAP = {
    "us_equity": "equity_index",
    "bonds": "bonds",
    "gold": "commodities",
    "metals": "commodities",
    "commodities": "commodities",
    "energy": "commodities",
    "agriculture": "commodities",
}

def _sector_of(ticker: str) -> str:
    grp = GROUP_MAP.get(ticker, DEFAULT_GROUP)
    return SECTOR_MAP.get(grp, "stock")


def _pct_change_over(prices: pd.Series, periods: int) -> pd.Series:
    return prices.pct_change(periods=periods)


def _rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=max(1, window // 2)).std()


def _rolling_max_drawdown(prices: pd.Series, window: int) -> pd.Series:
    roll_max = prices.rolling(window, min_periods=1).max()
    dd = (prices / roll_max) - 1.0
    return dd.rolling(window, min_periods=1).min()


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    mu = returns.rolling(window, min_periods=max(1, window // 2)).mean()
    sigma = returns.rolling(window, min_periods=max(1, window // 2)).std()
    return mu / (sigma + 1e-12)


def _rolling_skew(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=max(1, window // 2)).skew()


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0).rolling(window, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window, min_periods=1).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _mean_reversion_zscore(prices: pd.Series, window: int = 21) -> pd.Series:
    ma = prices.rolling(window, min_periods=max(1, window // 2)).mean()
    std = prices.rolling(window, min_periods=max(1, window // 2)).std()
    return (prices - ma) / (std + 1e-12)


def _gap_ratio(prices: pd.Series, window: int = 63) -> pd.Series:
    ret = prices.pct_change()
    vol = ret.rolling(window, min_periods=max(1, window // 2)).std()
    is_gap = ret.abs() > 2.0 * vol
    return is_gap.rolling(window, min_periods=1).mean()


def _relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    avg = volume.rolling(window, min_periods=max(1, window // 2)).mean()
    return volume / (avg + 1e-6)


def _volume_price_trend(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    ret = close.pct_change()
    vpt = (ret * volume).rolling(window, min_periods=1).sum()
    avg_vol = volume.rolling(window, min_periods=1).mean()
    return vpt / (avg_vol + 1e-6)


def _on_balance_volume_slope(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    direction = close.diff().apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    obv = (direction * volume).cumsum()
    return (obv - obv.shift(window)) / (window + 1e-12)


def _volume_volatility(volume: pd.Series, window: int = 21) -> pd.Series:
    log_vol = np.log1p(volume)
    return log_vol.rolling(window, min_periods=max(1, window // 2)).std()


def build_feature_frame(prices_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    has_volume = "volume" in df.columns
    out = []

    for tkr, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        px = g["close"].astype(float)
        ret1 = px.pct_change()

        g["mom_12_1"] = _pct_change_over(px.shift(21), 252)
        g["mom_6_1"] = _pct_change_over(px.shift(21), 126)
        g["mom_3_1"] = _pct_change_over(px.shift(21), 63)
        g["mom_1_0"] = _pct_change_over(px, 21)

        g["ma_50"] = px.rolling(50, min_periods=1).mean()
        g["ma_200"] = px.rolling(200, min_periods=1).mean()
        g["ma_200_ratio"] = px / (g["ma_200"] + 1e-12)
        g["trend_50_200"] = g["ma_50"] / (g["ma_200"] + 1e-12)

        g["vol_21"] = _rolling_vol(ret1, 21)
        g["vol_63"] = _rolling_vol(ret1, 63)
        g["vol_252"] = _rolling_vol(ret1, 252)
        g["maxdd_252"] = _rolling_max_drawdown(px, 252)
        g["maxdd_63"] = _rolling_max_drawdown(px, 63)

        g["sharpe_63"] = _rolling_sharpe(ret1, 63)
        g["sharpe_252"] = _rolling_sharpe(ret1, 252)

        g["mr_zscore_21"] = _mean_reversion_zscore(px, 21)
        g["rsi_14"] = _rsi(px, 14)

        g["skew_63"] = _rolling_skew(ret1, 63)
        g["gap_ratio_63"] = _gap_ratio(px, 63)
        g["vol_ratio_21_63"] = g["vol_21"] / (g["vol_63"] + 1e-12)

        if has_volume:
            vol = g["volume"].astype(float).fillna(0.0)
            g["rvol_20"] = _relative_volume(vol, 20)
            g["vpt_21"] = _volume_price_trend(px, vol, 21)
            g["obv_slope_21"] = _on_balance_volume_slope(px, vol, 21)
            g["vol_vol_21"] = _volume_volatility(vol, 21)
        else:
            g["rvol_20"] = np.nan
            g["vpt_21"] = np.nan
            g["obv_slope_21"] = np.nan
            g["vol_vol_21"] = np.nan

        g["sector"] = _sector_of(tkr)

        keep_cols = [
            "date", "ticker", "close", "sector",
            "mom_12_1", "mom_6_1", "mom_3_1", "mom_1_0",
            "ma_200_ratio", "trend_50_200",
            "vol_21", "vol_63", "vol_252",
            "maxdd_252", "maxdd_63",
            "sharpe_63", "sharpe_252",
            "mr_zscore_21", "rsi_14",
            "skew_63", "gap_ratio_63", "vol_ratio_21_63",
            "rvol_20", "vpt_21", "obv_slope_21", "vol_vol_21",
        ]
        out.append(g[keep_cols])

    feats = pd.concat(out, ignore_index=True)
    feats = _add_macro_features(feats)
    return feats


def _add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    macro_tickers = {
        "spy.us": "spy",
        "tlt.us": "tlt",
        "shy.us": "shy",
        "lqd.us": "lqd",
        "hyg.us": "hyg",
        "gld.us": "gld",
    }

    macro = {}
    for tkr, label in macro_tickers.items():
        sub = out[out["ticker"] == tkr][["date", "close"]].copy()
        sub = sub.drop_duplicates(subset="date").set_index("date").sort_index()
        sub = sub.rename(columns={"close": label})
        macro[label] = sub

    if not macro:
        for col in ["spy_mom_1m", "spy_vol_63", "spy_vol_21", "spy_vol_5", "yield_curve", "credit_spread", "gold_trend"]:
            out[col] = np.nan
        return out

    macro_df = pd.DataFrame(index=out["date"].drop_duplicates().sort_values())
    for label, sub in macro.items():
        macro_df = macro_df.join(sub, how="left")

    macro_df = macro_df.sort_index().ffill()

    if "spy" in macro_df.columns:
        macro_df["spy_mom_1m"] = macro_df["spy"].pct_change(21)
        spy_ret = macro_df["spy"].pct_change()
        # old signal
        macro_df["spy_vol_63"] = spy_ret.rolling(63, min_periods=10).std() * np.sqrt(252)
        # new "right now" signals
        macro_df["spy_vol_21"] = spy_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        macro_df["spy_vol_5"] = spy_ret.rolling(5, min_periods=5).std() * np.sqrt(252)
    else:
        macro_df["spy_mom_1m"] = np.nan
        macro_df["spy_vol_63"] = np.nan
        macro_df["spy_vol_21"] = np.nan
        macro_df["spy_vol_5"] = np.nan

    if "tlt" in macro_df.columns and "shy" in macro_df.columns:
        macro_df["yield_curve"] = macro_df["tlt"] / (macro_df["shy"] + 1e-12)
    else:
        macro_df["yield_curve"] = np.nan

    if "lqd" in macro_df.columns and "hyg" in macro_df.columns:
        macro_df["credit_spread"] = macro_df["lqd"] / (macro_df["hyg"] + 1e-12)
    else:
        macro_df["credit_spread"] = np.nan

    if "gld" in macro_df.columns:
        macro_df["gold_trend"] = macro_df["gld"].pct_change(63)
    else:
        macro_df["gold_trend"] = np.nan

    macro_cols = ["spy_mom_1m", "spy_vol_63", "spy_vol_21", "spy_vol_5", "yield_curve", "credit_spread", "gold_trend"]
    macro_out = macro_df[macro_cols].copy()

    out = out.merge(macro_out, left_on="date", right_index=True, how="left")
    return out


ZSCORE_COLS = [
    "mom_12_1", "mom_6_1", "mom_3_1", "mom_1_0",
    "ma_200_ratio", "trend_50_200",
    "vol_63", "vol_252", "maxdd_252", "maxdd_63",
    "sharpe_63", "sharpe_252",
    "mr_zscore_21", "rsi_14",
    "skew_63", "gap_ratio_63", "vol_ratio_21_63",
    "rvol_20", "vpt_21", "obv_slope_21", "vol_vol_21",
    "spy_mom_1m", "spy_vol_63", "spy_vol_21", "spy_vol_5",
    "yield_curve", "credit_spread", "gold_trend",
]

def add_cross_sectional_zscores(feats: pd.DataFrame) -> pd.DataFrame:
    df = feats.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ZSCORE_COLS:
        if c not in df.columns:
            continue
        df[c + "_z"] = df.groupby("date")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return df


def add_sector_relative_features(feats: pd.DataFrame) -> pd.DataFrame:
    df = feats.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "sector" not in df.columns:
        return df

    sector_rel_cols = ["mom_12_1", "mom_6_1", "mom_3_1", "sharpe_63", "vol_63", "ma_200_ratio"]
    for c in sector_rel_cols:
        if c not in df.columns:
            continue
        sector_mean = df.groupby(["date", "sector"])[c].transform("mean")
        sector_std = df.groupby(["date", "sector"])[c].transform("std")
        df[c + "_sec_rel"] = (df[c] - sector_mean) / (sector_std + 1e-12)
    return df