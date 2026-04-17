# src/pipeline.py
from __future__ import annotations

import pandas as pd
from typing import Optional

from .db import get_conn, init_db, get_last_date_for_ticker
from .stooq_data import fetch_daily_with_fallback
from .features import build_feature_frame, add_cross_sectional_zscores, add_sector_relative_features
from .model import make_monthly_recommendations
from .universe import get_universe
from .config import (
    CASH_TICKER,
    FINAL_MIN_POSITION_WEIGHT,
    PORTFOLIO_VALUE,
    MIN_TRADE_GBP,
    NO_TRADE_BAND,
    MAX_TURNOVER,
    TOTAL_K,
)
from .features_store import upsert_features
from .news_sentiment import update_news_sentiment
from .implication_news import update_implication_news
from .portfolio_tracking import get_current_portfolio_status, save_portfolio_snapshot, sync_latest_realised_performance


def _looks_like_connectivity_error(err: str | None) -> bool:
    if not err:
        return False
    e = err.lower()
    needles = (
        "connectionerror",
        "connecttimeout",
        "readtimeout",
        "timeout",
        "temporarily unavailable",
        "name resolution",
        "newconnectionerror",
        "failed to establish",
    )
    return any(n in e for n in needles)


def _iso_date(x) -> str:
    # SQLite-safe ISO string
    try:
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return str(x)


def upsert_prices(ticker: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    # Ensure SQLite-safe types
    df = df.copy()
    df["date"] = df["date"].apply(_iso_date)

    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                r["date"],
                ticker,
                float(r["open"]) if pd.notna(r.get("open")) else None,
                float(r["high"]) if pd.notna(r.get("high")) else None,
                float(r["low"]) if pd.notna(r.get("low")) else None,
                float(r.get("close")) if pd.notna(r.get("close")) else None,
                float(r.get("volume")) if pd.notna(r.get("volume")) else None,
            )
        )

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO prices_daily (date, ticker, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, ticker) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
            """,
            rows,
        )
    return len(rows)


def update_all_prices() -> None:
    universe = get_universe()
    any_rows = 0
    provider_counts = {"stooq": 0, "yahoo": 0}
    fail_count = 0
    successful_fetches = 0
    consecutive_connectivity_fails = 0
    skipped_due_to_outage = 0
    outage_threshold = 20

    for i, ticker in enumerate(universe):
        try:
            df, source, err = fetch_daily_with_fallback(ticker)
        except KeyboardInterrupt:
            fail_count += 1
            print(f"[WARN] Interrupted while fetching {ticker}; skipping ticker and continuing.")
            continue
        except Exception as e:
            fail_count += 1
            print(f"[WARN] Fetch crash for {ticker}: {type(e).__name__}: {e}")
            continue

        if df.empty:
            fail_count += 1
            reason = f" ({err})" if err else ""
            print(f"[WARN] No data for {ticker}{reason}")
            if _looks_like_connectivity_error(err):
                consecutive_connectivity_fails += 1
            else:
                consecutive_connectivity_fails = 0

            remaining = len(universe) - (i + 1)
            if successful_fetches == 0 and consecutive_connectivity_fails >= outage_threshold and remaining > 0:
                skipped_due_to_outage = remaining
                print(
                    "[WARN] Persistent provider connectivity failure detected; "
                    f"skipping remaining {remaining} tickers to avoid a multi-hour stall."
                )
                break
            continue

        successful_fetches += 1
        consecutive_connectivity_fails = 0
        provider_counts[source] = provider_counts.get(source, 0) + 1

        with get_conn() as conn:
            last_date = get_last_date_for_ticker(conn, ticker)

        if last_date is not None:
            # last_date is stored as text YYYY-MM-DD
            df = df[pd.to_datetime(df["date"]) > pd.to_datetime(last_date)]

        n = upsert_prices(ticker, df)
        any_rows += n
        with get_conn() as conn:
            up_to_date = get_last_date_for_ticker(conn, ticker)
        if n > 0:
            print(f"[OK] {ticker}: inserted or updated {n} rows (up to {up_to_date})")
        else:
            print(f"[OK] {ticker}: up to date ({up_to_date})")

    if any_rows == 0:
        print(
            "[WARN] No price rows inserted "
            f"(failures={fail_count}, fetched stooq={provider_counts.get('stooq', 0)}, "
            f"yahoo={provider_counts.get('yahoo', 0)}, "
            f"skipped_due_to_outage={skipped_due_to_outage})."
        )
    with get_conn() as conn:
        global_max_date = conn.execute("SELECT MAX(date) FROM prices_daily").fetchone()[0]
    print(f"[OK] prices_daily now up to {global_max_date}")


def load_prices_for_universe() -> pd.DataFrame:
    universe = get_universe()
    placeholders = ",".join(["?"] * len(universe))

    with get_conn() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT date, ticker, open, high, low, close, volume
            FROM prices_daily
            WHERE ticker IN ({placeholders})
            ORDER BY date ASC
            """,
            conn,
            params=universe,
        )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])

    # Synthetic cash series
    cash_dates = df["date"].drop_duplicates()
    cash = pd.DataFrame(
        {
            "date": cash_dates,
            "ticker": "cash",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 0.0,
        }
    )

    df = pd.concat([df, cash], ignore_index=True)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def _validate_recommendations(recs: pd.DataFrame) -> None:
    if recs.empty:
        raise ValueError("Recommendations dataframe is empty.")

    needed = {"asof_date", "ticker", "target_weight"}
    missing = needed.difference(set(recs.columns))
    if missing:
        raise ValueError(f"Recommendations missing required columns: {sorted(missing)}")

    x = recs.copy()
    x["asof_date"] = x["asof_date"].astype(str)
    x["ticker"] = x["ticker"].astype(str).str.lower()
    x["target_weight"] = pd.to_numeric(x["target_weight"], errors="coerce")

    if x["target_weight"].isna().any():
        bad = x[x["target_weight"].isna()].head(5)
        raise ValueError(f"NaN target_weight found in recommendations. Sample:\n{bad.to_string(index=False)}")
    if (x["target_weight"] < -1e-12).any():
        bad = x[x["target_weight"] < 0].head(5)
        raise ValueError(f"Negative target_weight found in recommendations. Sample:\n{bad.to_string(index=False)}")

    dups = x.duplicated(subset=["asof_date", "ticker"], keep=False)
    if dups.any():
        bad = x[dups].sort_values(["asof_date", "ticker"]).head(10)
        raise ValueError(f"Duplicate (asof_date, ticker) pairs found. Sample:\n{bad.to_string(index=False)}")

    sums = x.groupby("asof_date", as_index=False)["target_weight"].sum()
    bad_sums = sums[(sums["target_weight"] < 0.995) | (sums["target_weight"] > 1.005)]
    if not bad_sums.empty:
        raise ValueError(
            "Recommendation weights must sum to ~1.0 per asof_date.\n"
            + bad_sums.head(10).to_string(index=False)
        )

    active = x[x["target_weight"] > 1e-8]
    counts = active.groupby("asof_date")["ticker"].nunique()
    max_names = int(TOTAL_K) + 1  # +1 for optional cash row
    bad_counts = counts[counts > max_names]
    if not bad_counts.empty:
        sample = bad_counts.head(10)
        raise ValueError(
            f"Too many names in an asof_date snapshot (limit {max_names} incl cash).\n"
            + sample.to_string()
        )


def save_recommendations(recs: pd.DataFrame) -> int:
    if recs.empty:
        return 0

    # Ensure exactly one row per (asof_date, ticker), then fully replace each snapshot.
    recs = recs.drop_duplicates(subset=["asof_date", "ticker"], keep="last").copy()
    recs["asof_date"] = recs["asof_date"].astype(str)
    recs["ticker"] = recs["ticker"].astype(str).str.lower()

    _validate_recommendations(recs)

    rows = []
    for _, r in recs.iterrows():
        rows.append(
            (
                r["asof_date"],
                r["ticker"],
                r["action"],
                float(r["score"]),
                float(r["target_weight"]),
                r["reasons"],
            )
        )

    with get_conn() as conn:
        # Full table replacement keeps history internally consistent if month-end dates shift.
        conn.execute("DELETE FROM recommendations")
        conn.executemany(
            """
            INSERT INTO recommendations (asof_date, ticker, action, score, target_weight, reasons)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def _latest_close_map() -> dict:
    universe = get_universe()
    placeholders = ",".join(["?"] * len(universe))
    with get_conn() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT p1.ticker, p1.close
            FROM prices_daily p1
            JOIN (
                SELECT ticker, MAX(date) AS max_date
                FROM prices_daily
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
            ) p2
            ON p1.ticker = p2.ticker AND p1.date = p2.max_date
            """,
            conn,
            params=universe,
        )
    m = dict(zip(df["ticker"], df["close"]))
    m["cash"] = 1.0
    return m


def _apply_turnover_controls(w_current: dict, w_target: dict, band: float, max_turnover: float) -> dict:
    keys = set(w_current.keys()) | set(w_target.keys())
    wc = {k: w_current.get(k, 0.0) for k in keys}
    wt = {k: w_target.get(k, 0.0) for k in keys}

    for k in list(keys):
        if abs(wt[k] - wc[k]) < band:
            wt[k] = wc[k]

    s = sum(wt.values())
    if s > 0:
        wt = {k: v / s for k, v in wt.items()}

    turnover = sum(abs(wt[k] - wc[k]) for k in keys) / 2.0
    if turnover > max_turnover and turnover > 1e-12:
        scale = max_turnover / turnover
        wt = {k: wc[k] + (wt[k] - wc[k]) * scale for k in keys}
        s2 = sum(max(0.0, v) for v in wt.values())
        if s2 > 0:
            wt = {k: max(0.0, v) / s2 for k, v in wt.items()}

    return {k: v for k, v in wt.items() if v > 1e-6}


def _is_floor_constrained_position(ticker: str) -> bool:
    return str(ticker).lower() != CASH_TICKER


def _add_deficit(weights: dict, deficit: float) -> dict:
    if deficit <= 1e-12:
        return weights
    if CASH_TICKER in weights:
        weights[CASH_TICKER] = weights.get(CASH_TICKER, 0.0) + deficit
        return weights
    receivers = [k for k in weights if k == CASH_TICKER]
    if not receivers:
        receivers = list(weights.keys())
    base = sum(max(0.0, weights.get(k, 0.0)) for k in receivers)
    if base <= 1e-12:
        each = deficit / max(len(receivers), 1)
        for k in receivers:
            weights[k] = weights.get(k, 0.0) + each
        return weights
    for k in receivers:
        weights[k] = weights.get(k, 0.0) + deficit * (max(0.0, weights.get(k, 0.0)) / base)
    return weights


def _remove_excess(weights: dict, excess: float, locked: set[str]) -> dict:
    if excess <= 1e-12:
        return weights
    reducers = [k for k in weights if k not in locked and weights.get(k, 0.0) > 1e-12]
    room = sum(weights[k] for k in reducers)
    if room <= excess + 1e-12:
        total = sum(max(0.0, v) for v in weights.values())
        return {k: max(0.0, v) / total for k, v in weights.items()} if total > 0 else weights
    for k in reducers:
        weights[k] = max(0.0, weights[k] - excess * (weights[k] / room))
    return weights


def _enforce_final_position_floor(w_adjusted: dict, w_target: dict) -> dict:
    """
    The dashboard should show the chosen target portfolio, not tiny transition
    leftovers. Every non-cash holding below the floor is either chosen and
    lifted to the floor, or not chosen and fully exited.
    """
    chosen = {str(k).lower() for k, v in w_target.items() if float(v) > 1e-8}
    weights = {
        str(k).lower(): max(0.0, float(v))
        for k, v in w_adjusted.items()
        if str(k).lower() in chosen and float(v) > 1e-8
    }

    for ticker in chosen:
        if ticker not in weights:
            weights[ticker] = 0.0

    locked = set()
    for ticker in list(weights.keys()):
        if _is_floor_constrained_position(ticker) and weights[ticker] < FINAL_MIN_POSITION_WEIGHT:
            weights[ticker] = FINAL_MIN_POSITION_WEIGHT
            locked.add(ticker)

    total = sum(weights.values())
    if total <= 1e-12:
        return {}
    if total < 1.0:
        weights = _add_deficit(weights, 1.0 - total)
    elif total > 1.0:
        weights = _remove_excess(weights, total - 1.0, locked)

    total = sum(max(0.0, v) for v in weights.values())
    if total <= 1e-12:
        return {}
    return {k: max(0.0, v) / total for k, v in weights.items() if v > 1e-6}


def save_model_holdings_and_trades(portfolio_value: Optional[float] = None) -> None:
    trade_value = float(portfolio_value if portfolio_value is not None else PORTFOLIO_VALUE)
    with get_conn() as conn:
        latest_price = conn.execute("SELECT MAX(date) FROM prices_daily").fetchone()
        latest_price_date = latest_price[0] if latest_price and latest_price[0] else None
        if latest_price_date:
            cur = conn.execute(
                "SELECT MAX(asof_date) FROM recommendations WHERE asof_date <= ?",
                (latest_price_date,),
            )
        else:
            cur = conn.execute("SELECT MAX(asof_date) FROM recommendations")
        row = cur.fetchone()
        if not row or row[0] is None:
            return
        asof = row[0]

        recs = pd.read_sql_query(
            """
            SELECT ticker, target_weight
            FROM recommendations
            WHERE asof_date = ?
            """,
            conn,
            params=(asof,),
        )
        if recs.empty:
            return

        price_map = _latest_close_map()

        w_target = {}
        for _, r in recs.iterrows():
            tkr = str(r["ticker"]).lower()
            if tkr in price_map:
                w_target[tkr] = float(r["target_weight"])

        s = sum(w_target.values())
        if s > 0:
            w_target = {k: v / s for k, v in w_target.items()}

        cur2 = conn.execute(
            "SELECT MAX(asof_date) FROM model_holdings WHERE asof_date < ?",
            (asof,),
        )
        prev_row = cur2.fetchone()
        prev_asof = prev_row[0] if prev_row and prev_row[0] else None

        w_current = {}
        if prev_asof:
            prev_df = pd.read_sql_query(
                "SELECT ticker, target_weight FROM model_holdings WHERE asof_date = ?",
                conn,
                params=(prev_asof,),
            )
            if not prev_df.empty:
                w_current = dict(zip(prev_df["ticker"], prev_df["target_weight"].astype(float)))
                sc = sum(w_current.values())
                if sc > 0:
                    w_current = {k: v / sc for k, v in w_current.items()}

        w_adjusted = _apply_turnover_controls(w_current, w_target, NO_TRADE_BAND, MAX_TURNOVER)
        w_adjusted = _enforce_final_position_floor(w_adjusted, w_target)

        # Build holdings rows
        holdings_rows = []
        for tkr, w in w_adjusted.items():
            px = float(price_map.get(tkr, 0.0))
            if px <= 0:
                continue
            val = w * trade_value
            sh = val / px
            holdings_rows.append((asof, tkr, w, px, sh, val))

        conn.execute("DELETE FROM model_holdings WHERE asof_date = ?", (asof,))
        conn.executemany(
            """
            INSERT INTO model_holdings (asof_date, ticker, target_weight, price, shares, value)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(asof_date, ticker) DO UPDATE SET
                target_weight=excluded.target_weight,
                price=excluded.price,
                shares=excluded.shares,
                value=excluded.value
            """,
            holdings_rows,
        )

        # Trades
        prev_shares = {}
        if prev_asof:
            prev_sh_df = pd.read_sql_query(
                "SELECT ticker, shares FROM model_holdings WHERE asof_date = ?",
                conn,
                params=(prev_asof,),
            )
            if not prev_sh_df.empty:
                prev_shares = dict(zip(prev_sh_df["ticker"], prev_sh_df["shares"].astype(float)))

        trades_rows = []
        all_tickers = set(w_adjusted.keys()) | set(prev_shares.keys())
        for tkr in all_tickers:
            if tkr == CASH_TICKER:
                continue
            px = float(price_map.get(tkr, 0.0))
            if px <= 0:
                continue

            new_w = float(w_adjusted.get(tkr, 0.0))
            new_sh = (new_w * trade_value) / px
            old_sh = float(prev_shares.get(tkr, 0.0))
            delta = new_sh - old_sh
            if abs(delta) < 1e-9:
                continue

            est_notional = abs(delta) * px
            if est_notional < MIN_TRADE_GBP:
                continue

            action = "BUY" if delta > 0 else "SELL"
            trades_rows.append((asof, tkr, action, float(delta), float(est_notional)))

        conn.execute("DELETE FROM model_trades WHERE asof_date = ?", (asof,))
        if trades_rows:
            conn.executemany(
                """
                INSERT INTO model_trades (asof_date, ticker, trade_action, shares_delta, est_notional)
                VALUES (?, ?, ?, ?, ?)
                """,
                trades_rows,
            )

        turnover = sum(abs(w_adjusted.get(k, 0.0) - w_current.get(k, 0.0)) for k in all_tickers) / 2.0
        print(f"[OK] Trade plan portfolio value: £{trade_value:,.2f}")
        print(f"[OK] Turnover this month: {turnover*100:.1f}% (cap: {MAX_TURNOVER*100:.0f}%)")
        print(f"[OK] {len(trades_rows)} trades, {len(holdings_rows)} holdings")


def _resolve_portfolio_value(portfolio_value: Optional[float]) -> float:
    if portfolio_value is not None:
        resolved = float(portfolio_value)
        print(f"[OK] Using manual portfolio value override: £{resolved:,.2f}")
        return resolved

    status = get_current_portfolio_status()
    if status is not None and status.get("current_equity") is not None:
        resolved = float(status["current_equity"])
        print(
            "[OK] Inferred portfolio value from current prices: "
            f"£{resolved:,.2f} "
            f"({status['snapshot_asof_date']} -> {status['price_date']})"
        )
        return resolved

    print(f"[WARN] No saved portfolio snapshot found; using fallback seed value £{PORTFOLIO_VALUE:,.2f}")
    return float(PORTFOLIO_VALUE)


def run_pipeline(portfolio_value: Optional[float] = None) -> None:
    init_db()
    update_all_prices()

    # Live news sentiment
    try:
        news_features = update_news_sentiment()
        print(f"[OK] News sentiment: {len(news_features)} ticker scores")
    except Exception as e:
        print(f"[WARN] News sentiment failed (continuing without): {e}")
        news_features = None

    prices = load_prices_for_universe()
    if prices.empty:
        print("[ERROR] No prices found in DB")
        return
    current_asof_date = _iso_date(prices["date"].max())
    portfolio_value = _resolve_portfolio_value(portfolio_value)

    try:
        implication_features = update_implication_news(asof_date=current_asof_date)
        print(f"[OK] Implication news: {len(implication_features)} directional ticker scores")
    except Exception as e:
        print(f"[WARN] Implication news failed (continuing without): {e}")
        implication_features = None

    perf = sync_latest_realised_performance(current_asof_date)
    if perf is not None:
        print(
            "[OK] Performance computed "
            f"{perf['asof_date']} -> {perf['end_date']}: "
            f"£{perf['pnl_gbp']:.2f} ({perf['pnl_pct']*100:.2f}%)"
        )
        print(
            "[OK] Learning updated "
            f"for {perf['learning_updated']} tickers effective {current_asof_date}"
        )

    feats = build_feature_frame(prices)
    feats = add_cross_sectional_zscores(feats)
    feats = add_sector_relative_features(feats)

    # Merge news strictly on (date, ticker) to avoid look-ahead leakage.
    if news_features is not None and not news_features.empty:
        feats["date"] = pd.to_datetime(feats["date"])
        news_features["date"] = pd.to_datetime(news_features["date"])
        news_for_merge = news_features.copy()

        feats = feats.merge(news_for_merge, on=["ticker", "date"], how="left")
        for col in ["news_count_7d", "news_count_30d", "sent_mean_7d", "sent_mean_30d", "sent_shock"]:
            if col in feats.columns:
                feats[col] = feats[col].fillna(0.0)

        print(f"[OK] Merged news features for {len(news_for_merge)} tickers")

    if implication_features is not None and not implication_features.empty:
        feats["date"] = pd.to_datetime(feats["date"])
        implication_features["date"] = pd.to_datetime(implication_features["date"])
        feats = feats.merge(implication_features, on=["ticker", "date"], how="left")
        for col in ["implication_score_7d", "implication_score_30d", "implication_count_7d", "implication_count_30d"]:
            if col in feats.columns:
                feats[col] = feats[col].fillna(0.0)
        print(f"[OK] Merged implication features for {len(implication_features)} tickers")

    n_feats = upsert_features(feats)
    print(f"[OK] Saved {n_feats} feature rows")

    recs = make_monthly_recommendations(feats)
    _validate_recommendations(recs)
    n = save_recommendations(recs)
    print(f"[OK] Saved {n} recommendations")

    save_model_holdings_and_trades(portfolio_value=portfolio_value)
    print("[OK] Saved model holdings and trade plan")

    snapshot_asof = str(recs["asof_date"].max()) if not recs.empty else current_asof_date
    snapshot = save_portfolio_snapshot(snapshot_asof, starting_equity_override=portfolio_value)
    if snapshot is not None:
        print(
            "[OK] Snapshot saved "
            f"{snapshot['asof_date']}: "
            f"£{snapshot['starting_equity']:.2f} across {snapshot['holdings']} holdings"
        )
        if perf is not None and abs(float(snapshot["starting_equity"]) - float(perf["end_equity"])) > 1.0:
            print(
                "[OK] Reconciled snapshot to current portfolio value "
                f"£{portfolio_value:,.2f} (model mark-to-market was £{float(perf['end_equity']):,.2f})"
            )
