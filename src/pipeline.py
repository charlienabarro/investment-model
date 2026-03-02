import pandas as pd
from .db import get_conn, init_db, get_last_date_for_ticker
from .stooq_data import fetch_stooq_daily
from .features import build_feature_frame, add_cross_sectional_zscores, add_sector_relative_features
from .model import make_monthly_recommendations
from .universe import get_universe
from .config import PORTFOLIO_VALUE, MIN_TRADE_GBP, NO_TRADE_BAND, MAX_TURNOVER
from .features_store import upsert_features
from .news_sentiment import update_news_sentiment

def upsert_prices(ticker: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    rows = []
    for _, r in df.iterrows():
        rows.append((
            r["date"], ticker, r.get("open"), r.get("high"), r.get("low"), r.get("close"), r.get("volume")
        ))

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
            rows
        )
    return len(rows)

def update_all_prices() -> None:
    universe = get_universe()
    for ticker in universe:
        df = fetch_stooq_daily(ticker)
        if df.empty:
            print(f"[WARN] No data for {ticker}")
            continue

        with get_conn() as conn:
            last_date = get_last_date_for_ticker(conn, ticker)

        if last_date is not None:
            df = df[df["date"] > last_date]

        n = upsert_prices(ticker, df)
        print(f"[OK] {ticker}: inserted or updated {n} rows")

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
            params=universe
        )
    return df

def save_recommendations(recs: pd.DataFrame) -> int:
    if recs.empty:
        return 0

    rows = []
    for _, r in recs.iterrows():
        rows.append((
            r["asof_date"],
            r["ticker"],
            r["action"],
            float(r["score"]),
            float(r["target_weight"]),
            r["reasons"],
        ))

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO recommendations (asof_date, ticker, action, score, target_weight, reasons)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(asof_date, ticker) DO UPDATE SET
                action=excluded.action,
                score=excluded.score,
                target_weight=excluded.target_weight,
                reasons=excluded.reasons
            """,
            rows
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
            params=universe
        )
    return dict(zip(df["ticker"], df["close"]))

# Replace the entire save_model_holdings_and_trades function in pipeline.py with this.
# Also update the import line at the top of pipeline.py:
#
#   from .config import PORTFOLIO_VALUE, MIN_TRADE_GBP, NO_TRADE_BAND, MAX_TURNOVER
#

def _apply_turnover_controls(w_current: dict, w_target: dict, band: float, max_turnover: float) -> dict:
    """
    Smooth target weights against current holdings to reduce churn:
    1. No-trade band: if |target - current| < band, keep current weight
    2. Turnover cap: if total one-way turnover exceeds max, scale deltas down
    """
    keys = set(w_current.keys()) | set(w_target.keys())
    wc = {k: w_current.get(k, 0.0) for k in keys}
    wt = {k: w_target.get(k, 0.0) for k in keys}

    # No-trade band: ignore small weight changes
    for k in list(keys):
        if abs(wt[k] - wc[k]) < band:
            wt[k] = wc[k]

    # Re-normalise to 1.0
    s = sum(wt.values())
    if s > 0:
        wt = {k: v / s for k, v in wt.items()}

    # Turnover cap: scale deltas if total turnover is too high
    turnover = sum(abs(wt[k] - wc[k]) for k in keys) / 2.0
    if turnover > max_turnover and turnover > 1e-12:
        scale = max_turnover / turnover
        wt = {k: wc[k] + (wt[k] - wc[k]) * scale for k in keys}
        s2 = sum(max(0.0, v) for v in wt.values())
        if s2 > 0:
            wt = {k: max(0.0, v) / s2 for k, v in wt.items()}

    # Drop dust
    wt = {k: v for k, v in wt.items() if v > 1e-6}
    return wt


def save_model_holdings_and_trades() -> None:
    with get_conn() as conn:
        cur = conn.execute("SELECT MAX(asof_date) FROM recommendations")
        row = cur.fetchone()
        if not row or row[0] is None:
            return
        asof = row[0]

        # Load this month's raw target weights (only real holdings)
        recs = pd.read_sql_query(
            """
            SELECT ticker, target_weight
            FROM recommendations
            WHERE asof_date = ? AND action = 'BUY_OR_HOLD' AND target_weight >= 0.03
            """,
            conn,
            params=(asof,),
        )
        if recs.empty:
            return

        price_map = _latest_close_map()

        # Build raw target weight dict
        w_target = {}
        for _, r in recs.iterrows():
            tkr = r["ticker"]
            if tkr in price_map:
                w_target[tkr] = float(r["target_weight"])

        # Normalise target weights
        s = sum(w_target.values())
        if s > 0:
            w_target = {k: v / s for k, v in w_target.items()}

        # Load previous month's holdings to get current weights
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

        # Apply turnover controls: no-trade band + turnover cap
        w_adjusted = _apply_turnover_controls(w_current, w_target, NO_TRADE_BAND, MAX_TURNOVER)

        # Drop positions below minimum weight after smoothing
        w_adjusted = {k: v for k, v in w_adjusted.items() if v >= 0.03 and k in price_map}

        # Re-normalise
        s2 = sum(w_adjusted.values())
        if s2 > 0:
            w_adjusted = {k: v / s2 for k, v in w_adjusted.items()}

        # Build holdings rows using adjusted weights
        holdings_rows = []
        for tkr, w in w_adjusted.items():
            px = price_map[tkr]
            val = w * float(PORTFOLIO_VALUE)
            sh = val / px
            holdings_rows.append((asof, tkr, w, px, sh, val))

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

        # Build trades from adjusted weights vs previous weights
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
            px = price_map.get(tkr)
            if not px:
                continue

            new_w = w_adjusted.get(tkr, 0.0)
            new_sh = (new_w * float(PORTFOLIO_VALUE)) / px
            old_sh = prev_shares.get(tkr, 0.0)
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

        # Print summary
        turnover = sum(abs(w_adjusted.get(k, 0.0) - w_current.get(k, 0.0)) for k in all_tickers) / 2.0
        print(f"[OK] Turnover this month: {turnover*100:.1f}% (cap: {MAX_TURNOVER*100:.0f}%)")
        print(f"[OK] {len(trades_rows)} trades, {len(holdings_rows)} holdings")


def run_pipeline() -> None:
    init_db()
    update_all_prices()

    # ── NEW: Fetch and score news headlines ──
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

    feats = build_feature_frame(prices)
    feats = add_cross_sectional_zscores(feats)
    feats = add_sector_relative_features(feats)

    # ── NEW: Merge news features onto the latest date ──
    if news_features is not None and not news_features.empty:
        feats["date"] = pd.to_datetime(feats["date"])
        news_features["date"] = pd.to_datetime(news_features["date"])

        # News features only apply to the latest month-end snapshot
        # Merge on ticker, broadcast to the latest date in feats
        latest_date = feats["date"].max()
        news_for_merge = news_features.drop(columns=["date"]).copy()

        # Add news columns to all rows at the latest date
        feats = feats.merge(news_for_merge, on="ticker", how="left")

        # Fill NaN for tickers with no news
        for col in ["news_count_7d", "news_count_30d", "sent_mean_7d", "sent_mean_30d", "sent_shock"]:
            if col in feats.columns:
                feats[col] = feats[col].fillna(0.0)

        print(f"[OK] Merged news features for {len(news_for_merge)} tickers")

    # ---- SAVE FEATURES FOR STRICT BACKTESTING ----
    n_feats = upsert_features(feats)
    print(f"[OK] Saved {n_feats} feature rows")

    recs = make_monthly_recommendations(feats)
    n = save_recommendations(recs)
    print(f"[OK] Saved {n} recommendations")

    save_model_holdings_and_trades()
    print("[OK] Saved model holdings and trade plan")