# src/pipeline.py
from __future__ import annotations

import pandas as pd

from .db import get_conn, init_db, get_last_date_for_ticker
from .stooq_data import fetch_stooq_daily
from .features import build_feature_frame, add_cross_sectional_zscores
from .model import make_monthly_recommendations
from .universe import get_universe
from .config import PORTFOLIO_VALUE, BASE_DIR

from .sec_edgar import build_sec_filing_features
from .news_events import build_news_features


def upsert_prices(ticker: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    rows = []
    for _, r in df.iterrows():
        rows.append((
            r["date"], ticker, r.get("open"), r.get("high"), r.get("low"),
            r.get("close"), r.get("volume")
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

    with get_conn() as conn:
        # quick guard: if we already inserted prices today, don’t refetch everything
        today = pd.Timestamp.today().date().isoformat()
        already = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM prices_daily WHERE date = ?",
            conn,
            params=(today,),
        )["n"].iloc[0]
        if already > 0:
            print(f"[OK] Prices already present for today ({today}) — skipping Stooq fetch")
            return

    for ticker in universe:
        # only fetch small amount: last ~400 trading days
        df = fetch_stooq_daily(ticker)
        if df.empty:
            print(f"[WARN] No data for {ticker}")
            continue

        # keep only recent slice to reduce bandwidth + hits
        df = df.tail(450)

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
            SELECT date, ticker, close
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

    recs = recs.drop_duplicates(subset=["asof_date", "ticker"], keep="last")

    asof = recs["asof_date"].iloc[0]

    rows = []
    for _, r in recs.iterrows():
        rows.append((
            r["asof_date"], r["ticker"], r["action"],
            float(r["score"]), float(r["target_weight"]), r["reasons"]
        ))

    with get_conn() as conn:
        conn.execute("DELETE FROM recommendations WHERE asof_date = ?", (asof,))
        conn.executemany(
            """
            INSERT INTO recommendations (asof_date, ticker, action, score, target_weight, reasons)
            VALUES (?, ?, ?, ?, ?, ?)
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


def save_model_holdings_and_trades() -> None:
    with get_conn() as conn:
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
            params=(asof,)
        )
        if recs.empty:
            return

        price_map = _latest_close_map()
        recs["price"] = recs["ticker"].map(price_map).astype(float)
        recs = recs.dropna(subset=["price"])

        recs["target_value"] = recs["target_weight"].astype(float) * float(PORTFOLIO_VALUE)
        recs["shares"] = recs["target_value"] / recs["price"]
        recs["value"] = recs["shares"] * recs["price"]

        holdings_rows = []
        for _, r in recs.iterrows():
            holdings_rows.append((
                asof, r["ticker"],
                float(r["target_weight"]),
                float(r["price"]),
                float(r["shares"]),
                float(r["value"]),
            ))

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
            holdings_rows
        )

        cur2 = conn.execute(
            "SELECT MAX(asof_date) FROM model_holdings WHERE asof_date < ?",
            (asof,)
        )
        prev_row = cur2.fetchone()
        prev_asof = prev_row[0] if prev_row and prev_row[0] else None

        prev = pd.DataFrame(columns=["ticker", "shares"])
        if prev_asof:
            prev = pd.read_sql_query(
                "SELECT ticker, shares FROM model_holdings WHERE asof_date = ?",
                conn,
                params=(prev_asof,)
            )
        prev_map = dict(zip(prev["ticker"], prev["shares"])) if not prev.empty else {}

        trades_rows = []
        for _, r in recs.iterrows():
            tkr = r["ticker"]
            new_sh = float(r["shares"])
            old_sh = float(prev_map.get(tkr, 0.0))
            delta = new_sh - old_sh
            if abs(delta) < 1e-9:
                continue
            action = "BUY" if delta > 0 else "SELL"
            est_notional = abs(delta) * float(r["price"])
            trades_rows.append((asof, tkr, action, float(delta), float(est_notional)))

        conn.execute("DELETE FROM model_trades WHERE asof_date = ?", (asof,))
        if trades_rows:
            conn.executemany(
                """
                INSERT INTO model_trades (asof_date, ticker, trade_action, shares_delta, est_notional)
                VALUES (?, ?, ?, ?, ?)
                """,
                trades_rows
            )


def _enrich_with_external_features(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - SEC filings features (10-K, 10-Q, 8-K + days-since + counts)
      - News sentiment/volume + analyst event flags (upgrade/downgrade/PT changes)
    """
    if feats.empty:
        return feats

    feats = feats.copy()
    feats["date"] = pd.to_datetime(feats["date"])
    feats["ticker"] = feats["ticker"].astype(str).str.lower()

    tickers = sorted(feats["ticker"].unique().tolist())
    start = feats["date"].min()
    end = feats["date"].max()

    # SEC filings (free, high quality)
    sec_df = build_sec_filing_features(tickers, start_date=start, end_date=end)
    if not sec_df.empty:
        sec_df["date"] = pd.to_datetime(sec_df["date"])
        sec_df["ticker"] = sec_df["ticker"].astype(str).str.lower()
        feats = feats.merge(sec_df, on=["date", "ticker"], how="left")

    # News + “analyst event” detection via RSS headlines (free, patchy)
    news_df = build_news_features(tickers, start_date=start, end_date=end)
    if not news_df.empty:
        news_df["date"] = pd.to_datetime(news_df["date"])
        news_df["ticker"] = news_df["ticker"].astype(str).str.lower()
        feats = feats.merge(news_df, on=["date", "ticker"], how="left")

    # Fill missing numeric externals safely
    external_cols = [c for c in feats.columns if c.startswith(("filing_", "days_since_", "filings_", "sent_", "news_", "upgrades_", "downgrades_", "pt_"))]
    for c in external_cols:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)

    return feats


def run_pipeline() -> None:
    init_db()
    update_all_prices()

    prices = load_prices_for_universe()
    if prices.empty:
        print("[ERROR] No prices found in DB")
        return

    feats = build_feature_frame(prices)

    feats = _enrich_with_external_features(feats)

    feats = add_cross_sectional_zscores(feats)

    recs = make_monthly_recommendations(feats)
    n = save_recommendations(recs)
    print(f"[OK] Saved {n} recommendations")

    save_model_holdings_and_trades()
    print("[OK] Saved model holdings and trade plan")