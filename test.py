#!/usr/bin/env python3
"""
test.py — realistic monthly backtest (Trading212 style)

What it simulates:
- Once per month: get target weights (from rerunning your current model on historical features, if available)
  otherwise falls back to DB-stored recommendations.
- Execute rebalance on next available trading day after asof_date (close-to-close).
- Between rebalances: hold positions, portfolio value changes daily with prices.
- Trade only the deltas (not "sell everything"), with:
  - no-trade band (small weight differences are ignored)
  - max turnover cap (optional)
- Costs:
  - Trading212 commission: 0
  - FX fee: default 0.15% on traded notional for USD instruments (tickers ending in ".us")
  - optional slippage bps (default 0)

Important limitations:
- We assume your prices are in one consistent currency already in DB.
  FX fee is applied as a % of traded notional in those instruments (approx realistic for GBP accounts trading USD).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.db import get_conn
from src.model import make_monthly_recommendations


# ---- Trading controls (align with “realistic minimal thinking”) ----
NO_TRADE_BAND = 0.02     # ignore weight changes smaller than 2%
MAX_TURNOVER = 0.30      # cap total one-way turnover at 30% per rebalance


@dataclass
class Stats:
    final_equity: float
    cagr: float
    vol: float
    sharpe: float
    max_drawdown: float


def _dt(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def _is_usd_instrument(ticker: str) -> bool:
    # Your universe uses stooq style: e.g. spy.us, aapl.us
    return ticker.lower().endswith(".us")


def _table_exists(conn, name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    r = pd.read_sql_query(q, conn, params=(name,))
    return not r.empty


def _detect_features_table(conn) -> Optional[str]:
    # Try common names
    candidates = ["features_daily", "features", "signals", "model_features"]
    for c in candidates:
        if _table_exists(conn, c):
            return c
    return None


def _load_prices(conn) -> pd.DataFrame:
    # Requires prices_daily(ticker, date, close)
    px = pd.read_sql_query(
        "SELECT ticker, date, close FROM prices_daily",
        conn
    )
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    return px


def _next_trading_date(global_dates: List[pd.Timestamp], after: pd.Timestamp) -> Optional[pd.Timestamp]:
    # global_dates is sorted unique trading dates
    # return first date > after
    import bisect
    i = bisect.bisect_right(global_dates, after)
    if i >= len(global_dates):
        return None
    return global_dates[i]


def _max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min())


def _compute_stats(equity_curve: pd.DataFrame) -> Stats:
    eq = equity_curve["equity"].astype(float)
    dates = pd.to_datetime(equity_curve["date"])

    final_equity = float(eq.iloc[-1])

    rets = eq.pct_change().dropna()
    if len(rets) < 3:
        return Stats(final_equity=final_equity, cagr=float("nan"), vol=float("nan"), sharpe=float("nan"), max_drawdown=_max_drawdown(eq))

    # annualisation based on daily points
    vol = float(rets.std(ddof=0) * math.sqrt(252))
    mean = float(rets.mean() * 252)
    sharpe = mean / vol if vol > 1e-12 else float("nan")

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    start = float(eq.iloc[0])
    cagr = (final_equity / start) ** (1.0 / years) - 1.0 if years > 0 and start > 0 else float("nan")

    mdd = _max_drawdown(eq)

    return Stats(final_equity=final_equity, cagr=cagr, vol=vol, sharpe=sharpe, max_drawdown=mdd)


def _weights_from_db(conn, asof_date: str) -> Dict[str, float]:
    df = pd.read_sql_query(
        """
        SELECT ticker, target_weight, action
        FROM recommendations
        WHERE asof_date = ?
        """,
        conn,
        params=(asof_date,)
    )
    df["target_weight"] = df["target_weight"].astype(float)
    df = df[(df["action"] == "BUY_OR_HOLD") & (df["target_weight"] > 0)].copy()

    w = {r["ticker"]: float(r["target_weight"]) for _, r in df.iterrows()}
    s = sum(w.values())
    if s > 0:
        w = {k: v / s for k, v in w.items()}
    return w


def _weights_rerun_model(conn, features_table: str, asof_date: str, prices_df: pd.DataFrame = None) -> Dict[str, float]:
    cutoff = _dt(asof_date)

    feats = pd.read_sql_query(
        f"SELECT * FROM {features_table} WHERE date <= ?",
        conn,
        params=(cutoff.date().isoformat(),)
    )
    if feats.empty:
        return {}

    # model needs a 'close' column for correlation filtering
    if "close" not in feats.columns:
        feats["date"] = pd.to_datetime(feats["date"])
        if prices_df is not None:
            px = prices_df[prices_df["date"] <= cutoff].copy()
        else:
            px = pd.read_sql_query(
                "SELECT date, ticker, close FROM prices_daily WHERE date <= ?",
                conn,
                params=(cutoff.date().isoformat(),)
            )
        feats = feats.merge(px[["date", "ticker", "close"]], on=["date", "ticker"], how="left")

    recs = make_monthly_recommendations(feats)

    recs = recs[recs["asof_date"] == cutoff.date().isoformat()].copy()
    recs = recs[(recs["action"] == "BUY_OR_HOLD") & (recs["target_weight"] > 0)].copy()

    w = {r["ticker"]: float(r["target_weight"]) for _, r in recs.iterrows()}
    s = sum(w.values())
    if s > 0:
        w = {k: v / s for k, v in w.items()}
    return w


def _current_weights(positions: Dict[str, float], prices_today: Dict[str, float]) -> Dict[str, float]:
    """
    Convert share positions to weights at today's prices.
    """
    values = {}
    total = 0.0
    for t, sh in positions.items():
        p = prices_today.get(t)
        if p is None or p <= 0:
            continue
        v = float(sh) * float(p)
        values[t] = v
        total += v

    if total <= 0:
        return {}

    return {t: v / total for t, v in values.items()}


def _apply_no_trade_band_and_turnover_cap(
    w_current: Dict[str, float],
    w_target: Dict[str, float],
    band: float,
    max_turnover: float
) -> Dict[str, float]:
    """
    Adjust targets to reduce churn:
    - if |target - current| < band, keep current weight
    - cap one-way turnover (sum |delta| / 2) to max_turnover by scaling deltas
    """
    keys = set(w_current.keys()) | set(w_target.keys())
    wc = {k: w_current.get(k, 0.0) for k in keys}
    wt = {k: w_target.get(k, 0.0) for k in keys}

    # no-trade band
    for k in list(keys):
        if abs(wt[k] - wc[k]) < band:
            wt[k] = wc[k]

    # normalise target to 1.0
    s = sum(wt.values())
    if s > 0:
        wt = {k: v / s for k, v in wt.items()}

    # turnover cap
    turnover = sum(abs(wt[k] - wc[k]) for k in keys) / 2.0
    if turnover > max_turnover and turnover > 1e-12:
        scale = max_turnover / turnover
        wt = {k: wc[k] + (wt[k] - wc[k]) * scale for k in keys}
        # renormalise again
        s2 = sum(max(0.0, v) for v in wt.values())
        if s2 > 0:
            wt = {k: max(0.0, v) / s2 for k, v in wt.items()}

    # drop tiny float noise
    wt = {k: v for k, v in wt.items() if v > 1e-12}

    return wt


def _rebalance(
    portfolio_value: float,
    w_target: Dict[str, float],
    prices_today: Dict[str, float],
) -> Dict[str, float]:
    """
    Convert target weights to new share positions at today's close.
    """
    positions = {}
    for t, w in w_target.items():
        p = prices_today.get(t)
        if p is None or p <= 0:
            continue
        positions[t] = (portfolio_value * float(w)) / float(p)
    return positions


def _trade_cost(
    portfolio_value: float,
    w_current: Dict[str, float],
    w_target: Dict[str, float],
    fx_fee: float,
    slippage_bps: float,
) -> float:
    """
    Compute cost as a fraction of portfolio value based on traded notional:
    cost = sum(|delta_w|) * (fx_fee_if_usd + slippage)
    Here we approximate notional traded as portfolio_value * |delta_w|.
    FX fee applied only for USD instruments.
    """
    keys = set(w_current.keys()) | set(w_target.keys())

    slip = float(slippage_bps) / 10_000.0
    total_cost = 0.0

    for k in keys:
        dw = abs(w_target.get(k, 0.0) - w_current.get(k, 0.0))
        if dw <= 0:
            continue

        # slippage always applies
        c = dw * slip

        # FX fee applies for USD instruments on traded notional
        if _is_usd_instrument(k):
            c += dw * float(fx_fee)

        total_cost += c

    # total_cost is fraction of portfolio value
    return float(total_cost)


def run_backtest(
    start_asof: str,
    initial_value: float,
    fx_fee: float,
    slippage_bps: float,
    mode: str,
    out_csv: str,
) -> Stats:
    with get_conn() as conn:
        prices = _load_prices(conn)

        # global trading calendar
        all_dates = sorted(prices["date"].dropna().unique().tolist())
        all_dates = [pd.Timestamp(d).normalize() for d in all_dates]

        # available recommendation months
        asof_df = pd.read_sql_query(
            "SELECT DISTINCT asof_date FROM recommendations ORDER BY asof_date ASC",
            conn
        )
        all_asofs = asof_df["asof_date"].astype(str).tolist()
        all_asofs = [d for d in all_asofs if d >= start_asof]

        if len(all_asofs) < 2:
            raise RuntimeError("Not enough monthly snapshots in recommendations table. Run run_pipeline.py first.")

        features_table = _detect_features_table(conn)
        if mode == "rerun" and not features_table:
            print("[WARN] No features table found in DB; falling back to DB recommendations.")
            mode = "db"

        # helper: prices for a given date as dict
        prices_by_date = {}
        for dt, g in prices.groupby("date"):
            prices_by_date[pd.Timestamp(dt).normalize()] = {r["ticker"]: float(r["close"]) for _, r in g.iterrows()}

        equity_rows = []

        # positions live in shares
        positions: Dict[str, float] = {}
        portfolio_value = float(initial_value)

        # we simulate daily equity values, but only rebalance monthly
        # pick first rebalance trade date: next trading day after first asof
        for i, asof in enumerate(all_asofs):
            asof_dt = _dt(asof)
            trade_dt = _next_trading_date(all_dates, asof_dt)
            if trade_dt is None:
                break

            # Determine next rebalance date for holding period end
            next_asof_dt = _dt(all_asofs[i + 1]) if i + 1 < len(all_asofs) else None
            end_dt = _next_trading_date(all_dates, next_asof_dt) if next_asof_dt is not None else None
            if end_dt is None:
                # no future trade day, stop after valuing up to last available date
                end_dt = all_dates[-1]

            # choose target weights for this asof
            if mode == "rerun":
                print(f"[{i + 1}/{len(all_asofs)}] Rebalancing {asof}...")
                w_target = _weights_rerun_model(conn, features_table, asof, prices_df=prices)
            else:
                w_target = _weights_from_db(conn, asof)

            if not w_target:
                continue

            # price dict on trade date
            px_trade = prices_by_date.get(trade_dt, {})
            # mark-to-market current portfolio at trade date before rebalancing
            if positions:
                portfolio_value = sum(positions.get(t, 0.0) * px_trade.get(t, 0.0) for t in positions.keys())

            # compute churn controls vs current weights
            w_current = _current_weights(positions, px_trade) if positions else {}
            w_adj = _apply_no_trade_band_and_turnover_cap(w_current, w_target, NO_TRADE_BAND, MAX_TURNOVER)

            # apply costs on the traded deltas
            cost_frac = _trade_cost(portfolio_value, w_current, w_adj, fx_fee=fx_fee, slippage_bps=slippage_bps)
            portfolio_value *= (1.0 - cost_frac)

            # rebalance to adjusted target weights
            positions = _rebalance(portfolio_value, w_adj, px_trade)

            # Now simulate daily equity until the day before end_dt (we rebalance on end_dt)
            # Include trade_dt in curve.
            holding_dates = [d for d in all_dates if trade_dt <= d < end_dt]
            for d in holding_dates:
                px = prices_by_date.get(d, {})
                equity = sum(positions.get(t, 0.0) * px.get(t, 0.0) for t in positions.keys())
                equity_rows.append({
                    "date": d.date().isoformat(),
                    "equity": float(equity),
                    "asof_date": asof,
                    "trade_date": trade_dt.date().isoformat(),
                    "cost_frac": float(cost_frac),
                    "mode": mode,
                })

            # Update portfolio_value at end of holding period
            last_day = holding_dates[-1] if holding_dates else trade_dt
            px_last = prices_by_date.get(last_day, {})
            portfolio_value = sum(positions.get(t, 0.0) * px_last.get(t, 0.0) for t in positions.keys())

            if end_dt == all_dates[-1]:
                break

        curve = pd.DataFrame(equity_rows)
        if curve.empty:
            raise RuntimeError("Equity curve is empty. Check start date and DB contents.")

        curve = curve.drop_duplicates(subset=["date"]).sort_values("date")
        curve.to_csv(out_csv, index=False)

        stats = _compute_stats(curve)
        return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="Start asof_date, e.g. 2020-01-31")
    parser.add_argument("--initial", type=float, default=10000.0, help="Initial portfolio value")
    parser.add_argument("--mode", choices=["auto", "rerun", "db"], default="auto",
                        help="auto tries rerun model on features table if available, else DB recommendations")
    parser.add_argument("--fx-fee", type=float, default=0.0015,
                        help="FX fee rate (Trading212 GBP default is 0.0015 for 0.15%)")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Extra slippage in bps on traded notional")
    parser.add_argument("--out", type=str, default="equity_curve.csv", help="Output equity curve CSV")
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        # decide at runtime
        with get_conn() as conn:
            ft = _detect_features_table(conn)
        mode = "rerun" if ft else "db"

    stats = run_backtest(
        start_asof=args.start,
        initial_value=args.initial,
        fx_fee=args.fx_fee,
        slippage_bps=args.slippage_bps,
        mode=mode,
        out_csv=args.out,
    )

    print("Backtest complete")
    print(f"Final equity: {stats.final_equity:.2f}")
    print(f"CAGR: {stats.cagr*100:.2f}%")
    print(f"Vol: {stats.vol*100:.2f}%")
    print(f"Sharpe: {stats.sharpe:.3f}")
    print(f"Max drawdown: {stats.max_drawdown*100:.2f}%")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()