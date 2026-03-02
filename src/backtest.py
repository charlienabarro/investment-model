import numpy as np
import pandas as pd


def compute_perf_stats(equity: pd.Series, trading_days: int = 252) -> dict:
    equity = equity.dropna()
    if len(equity) < 3:
        return {"cagr": None, "vol": None, "sharpe": None, "max_drawdown": None}

    rets = equity.pct_change().dropna()
    if rets.empty:
        return {"cagr": None, "vol": None, "sharpe": None, "max_drawdown": None}

    start = equity.index[0]
    end = equity.index[-1]
    years = (end - start).days / 365.25
    if years <= 0:
        return {"cagr": None, "vol": None, "sharpe": None, "max_drawdown": None}

    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    vol = float(rets.std() * np.sqrt(trading_days))
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * np.sqrt(trading_days))

    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    max_dd = float(dd.min())

    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_drawdown": max_dd}


def run_recommendation_backtest(
    prices: pd.DataFrame,
    recs: pd.DataFrame,
    initial_capital: float = 1.0,
    cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """
    prices columns: date, ticker, close
    recs columns: asof_date, ticker, target_weight

    Logic:
    - At each rebalance signal date (asof_date), apply weights from next trading day onward
    - Hold until next rebalance effective date
    - Apply simple transaction cost on turnover at rebalances (cost_bps in basis points)
    """
    if prices.empty or recs.empty:
        out = pd.DataFrame(columns=["date", "equity", "daily_return", "turnover"])
        stats = compute_perf_stats(pd.Series(dtype=float))
        return out, stats

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["date", "ticker"])

    r = recs.copy()
    r["asof_date"] = pd.to_datetime(r["asof_date"])
    r = r.sort_values(["asof_date", "ticker"])

    px = p.pivot(index="date", columns="ticker", values="close").sort_index()
    ret = px.pct_change().fillna(0.0)

    # Build schedule of target weights at each asof_date
    w_by_date = {}
    for d, g in r.groupby("asof_date"):
        g = g.copy()
        g = g[g["target_weight"] > 0].copy()
        if g.empty:
            w_by_date[d] = {}
            continue
        weights = dict(zip(g["ticker"], g["target_weight"]))
        s = sum(weights.values())
        if s > 0:
            weights = {k: v / s for k, v in weights.items()}
        w_by_date[d] = weights

    signal_dates = sorted(w_by_date.keys())
    if not signal_dates:
        out = pd.DataFrame(columns=["date", "equity", "daily_return", "turnover"])
        stats = compute_perf_stats(pd.Series(dtype=float))
        return out, stats

    # Map each asof_date to the next trading day in price index to avoid lookahead
    trading_index = px.index

    def next_trading_day(d: pd.Timestamp) -> pd.Timestamp | None:
        pos = trading_index.searchsorted(d)
        if pos >= len(trading_index):
            return None
        # if asof_date is itself a trading day, executing next day is safer
        if trading_index[pos] == d:
            pos += 1
        if pos >= len(trading_index):
            return None
        return trading_index[pos]

    effective = []
    for sd in signal_dates:
        nd = next_trading_day(sd)
        if nd is not None:
            effective.append((sd, nd))
    if not effective:
        out = pd.DataFrame(columns=["date", "equity", "daily_return", "turnover"])
        stats = compute_perf_stats(pd.Series(dtype=float))
        return out, stats

    # Create a daily weight matrix
    weights_daily = pd.DataFrame(0.0, index=trading_index, columns=px.columns)

    for i, (sd, eff) in enumerate(effective):
        start = eff
        end = effective[i + 1][1] if i + 1 < len(effective) else trading_index[-1] + pd.Timedelta(days=1)

        w = w_by_date.get(sd, {})
        if w:
            for tkr, val in w.items():
                if tkr in weights_daily.columns:
                    weights_daily.loc[(weights_daily.index >= start) & (weights_daily.index < end), tkr] = val

    # Portfolio daily returns
    port_ret = (weights_daily.shift(1).fillna(0.0) * ret).sum(axis=1)

    # Transaction costs on rebalance effective dates
    turnover = pd.Series(0.0, index=trading_index)
    prev_w = pd.Series(0.0, index=px.columns)

    for sd, eff in effective:
        cur_w = weights_daily.loc[eff].fillna(0.0)
        tv = float((cur_w - prev_w).abs().sum())
        turnover.loc[eff] = tv
        prev_w = cur_w

    cost = turnover * (cost_bps / 10000.0)
    net_ret = port_ret - cost

    equity = (1.0 + net_ret).cumprod() * initial_capital

    out = pd.DataFrame({
        "date": trading_index,
        "equity": equity.values,
        "daily_return": net_ret.values,
        "turnover": turnover.values,
    })

    stats = compute_perf_stats(equity)
    return out, stats