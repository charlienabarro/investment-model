# src/api.py — V3 with improved dashboard
import json
import io
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from .db import get_conn, init_db
from .universe import get_universe
from .backtest import run_recommendation_backtest
from .config import BACKTEST_START_DATE, CASH_TICKER, PORTFOLIO_VALUE
from .portfolio_tracking import get_current_portfolio_status, get_portfolio_history, get_portfolio_performance
from .commodity_scout import COMMODITY_UNIVERSE, run_commodity_backtest

app = FastAPI(title="Investing Backend")


def _latest_price_date(conn):
    row = conn.execute("SELECT MAX(date) FROM prices_daily").fetchone()
    return row[0] if row and row[0] else None


def _latest_asof_on_or_before_prices(conn, table: str):
    latest_price_date = _latest_price_date(conn)
    if latest_price_date:
        row = conn.execute(
            f"SELECT MAX(asof_date) FROM {table} WHERE asof_date <= ?",
            (latest_price_date,),
        ).fetchone()
    else:
        row = conn.execute(f"SELECT MAX(asof_date) FROM {table}").fetchone()
    return row[0] if row and row[0] else None


def _recent_recommendation_dates(conn, limit: int) -> pd.DataFrame:
    return _recent_asof_dates(conn, "recommendations", limit)


def _recent_asof_dates(conn, table: str, limit: int) -> pd.DataFrame:
    try:
        limit = int(limit)
    except TypeError:
        limit = 2

    latest_price_date = _latest_price_date(conn)
    if latest_price_date:
        return pd.read_sql_query(
            f"""
            SELECT DISTINCT asof_date
            FROM {table}
            WHERE asof_date <= ?
            ORDER BY asof_date DESC
            LIMIT ?
            """,
            conn,
            params=(latest_price_date, limit),
        )
    return pd.read_sql_query(
        f"SELECT DISTINCT asof_date FROM {table} ORDER BY asof_date DESC LIMIT ?",
        conn,
        params=(limit,),
    )


def _current_portfolio_value() -> float:
    status = get_current_portfolio_status()
    if status is not None and status.get("current_equity") is not None:
        return float(status["current_equity"])
    perf = get_portfolio_performance()
    if perf is not None and perf.get("current_equity") is not None:
        return float(perf["current_equity"])
    return float(PORTFOLIO_VALUE)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/universe")
def universe():
    return {"tickers": get_universe()}


@app.get("/prices/latest")
def latest_prices():
    universe = get_universe()
    placeholders = ",".join(["?"] * len(universe))
    with get_conn() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT p1.ticker, p1.date, p1.close
            FROM prices_daily p1
            JOIN (
                SELECT ticker, MAX(date) AS max_date
                FROM prices_daily
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
            ) p2
            ON p1.ticker = p2.ticker AND p1.date = p2.max_date
            ORDER BY p1.ticker ASC
            """,
            conn,
            params=universe,
        )
    return df.to_dict(orient="records")


@app.get("/recommendations/latest")
def latest_recommendations():
    with get_conn() as conn:
        asof = _latest_asof_on_or_before_prices(conn, "recommendations")
        if asof is None:
            return {"asof_date": None, "recs": []}
        df = pd.read_sql_query(
            """
            SELECT asof_date, ticker, action, score, target_weight, reasons
            FROM recommendations
            WHERE asof_date = ? AND target_weight > 0
            ORDER BY target_weight DESC, score DESC, ticker ASC
            """,
            conn,
            params=(asof,),
        )
    return {"asof_date": asof, "recs": df.to_dict(orient="records")}


@app.get("/recommendations/history")
def recommendations_history(limit: int = Query(2, ge=1, le=24)):
    with get_conn() as conn:
        dates = _recent_recommendation_dates(conn, limit)
        out = []
        for asof in dates["asof_date"].tolist():
            df = pd.read_sql_query(
                """
                SELECT asof_date, ticker, action, score, target_weight, reasons
                FROM recommendations
                WHERE asof_date = ? AND target_weight > 0
                ORDER BY target_weight DESC, score DESC, ticker ASC
                """,
                conn,
                params=(asof,),
            )
            out.append({"asof_date": asof, "recs": df.to_dict(orient="records")})
    return {"snapshots": out}


def _model_holdings_frame(conn, asof: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            h.asof_date,
            h.ticker,
            COALESCE(r.action, CASE WHEN h.ticker = 'cash' THEN 'HOLD_CASH' ELSE 'BUY_OR_HOLD' END) AS action,
            COALESCE(r.score, 0.0) AS score,
            h.target_weight,
            h.price,
            h.shares,
            h.value,
            COALESCE(
                r.reasons,
                CASE
                    WHEN h.ticker = 'cash' THEN 'Cash reserve carried in the turnover-controlled target portfolio.'
                    ELSE 'Carried in the turnover-controlled target portfolio while changes are phased in.'
                END
            ) AS reasons
        FROM model_holdings h
        LEFT JOIN recommendations r
          ON r.asof_date = h.asof_date
         AND r.ticker = h.ticker
        WHERE h.asof_date = ? AND h.target_weight > 0
        ORDER BY h.target_weight DESC, h.ticker ASC
        """,
        conn,
        params=(asof,),
    )


@app.get("/model/holdings/latest")
def latest_model_holdings():
    with get_conn() as conn:
        asof = _latest_asof_on_or_before_prices(conn, "model_holdings")
        if asof is None:
            return latest_recommendations()
        df = _model_holdings_frame(conn, asof)
    return {"asof_date": asof, "recs": df.to_dict(orient="records")}


@app.get("/model/holdings/history")
def model_holdings_history(limit: int = Query(2, ge=1, le=24)):
    with get_conn() as conn:
        dates = _recent_asof_dates(conn, "model_holdings", limit)
        out = []
        for asof in dates["asof_date"].tolist():
            df = _model_holdings_frame(conn, asof)
            out.append({"asof_date": asof, "recs": df.to_dict(orient="records")})
    return {"snapshots": out}


@app.get("/model/trades/latest")
def latest_model_trades():
    with get_conn() as conn:
        asof = _latest_asof_on_or_before_prices(conn, "model_trades")
        if asof is None:
            return {"asof_date": None, "trades": []}
        df = pd.read_sql_query(
            """
            SELECT asof_date, ticker, trade_action, shares_delta, est_notional
            FROM model_trades
            WHERE asof_date = ?
            ORDER BY est_notional DESC, ticker ASC
            """,
            conn,
            params=(asof,),
        )
    return {"asof_date": asof, "trades": df.to_dict(orient="records")}


def _commodity_holdings_frame(conn, asof: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            asof_date,
            ticker,
            commodity,
            CASE WHEN ticker = 'cash' THEN 'HOLD_CASH' ELSE 'BUY_OR_HOLD' END AS action,
            confidence,
            target_weight,
            invested_weight,
            price,
            shares,
            value,
            trading212_name,
            trading212_ticker,
            reasons
        FROM commodity_model_holdings
        WHERE asof_date = ? AND target_weight > 0
        ORDER BY target_weight DESC, ticker ASC
        """,
        conn,
        params=(asof,),
    )


def _commodity_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_weight": 0.0,
            "invested_pie_weight": 0.0,
            "cash_weight": 0.0,
            "cash_value": 0.0,
            "positions": 0,
            "non_cash_positions": 0,
        }
    total = float(df["target_weight"].astype(float).sum())
    cash = df[df["ticker"].astype(str).str.lower() == "cash"]
    noncash = df[df["ticker"].astype(str).str.lower() != "cash"]
    return {
        "total_weight": total,
        "invested_pie_weight": float(noncash["invested_weight"].astype(float).sum()) if not noncash.empty else 0.0,
        "cash_weight": float(cash["target_weight"].astype(float).sum()) if not cash.empty else 0.0,
        "cash_value": float(cash["value"].astype(float).sum()) if not cash.empty else 0.0,
        "positions": int(len(df)),
        "non_cash_positions": int(len(noncash)),
        "min_non_cash_weight": float(noncash["target_weight"].astype(float).min()) if not noncash.empty else 0.0,
        "max_non_cash_weight": float(noncash["target_weight"].astype(float).max()) if not noncash.empty else 0.0,
    }


@app.get("/commodities/latest")
def commodities_latest():
    with get_conn() as conn:
        asof = _latest_asof_on_or_before_prices(conn, "commodity_model_holdings")
        if asof is None:
            return {"asof_date": None, "summary": _commodity_summary(pd.DataFrame()), "holdings": []}
        df = _commodity_holdings_frame(conn, asof)
    return {"asof_date": asof, "summary": _commodity_summary(df), "holdings": df.to_dict(orient="records")}


@app.get("/commodities/history")
def commodities_history(limit: int = Query(6, ge=1, le=24)):
    with get_conn() as conn:
        dates = _recent_asof_dates(conn, "commodity_model_holdings", limit)
        out = []
        for asof in dates["asof_date"].tolist():
            df = _commodity_holdings_frame(conn, asof)
            out.append({"asof_date": asof, "summary": _commodity_summary(df), "holdings": df.to_dict(orient="records")})
    return {"snapshots": out}


@app.get("/commodities/news")
def commodities_news(limit: int = Query(40, ge=1, le=200)):
    with get_conn() as conn:
        feature_asof = conn.execute("SELECT MAX(date) FROM commodity_news_features").fetchone()
        asof = feature_asof[0] if feature_asof and feature_asof[0] else None
        features = pd.DataFrame()
        if asof:
            features = pd.read_sql_query(
                """
                SELECT *
                FROM commodity_news_features
                WHERE date = ?
                ORDER BY news_count_7d DESC, sent_shock DESC, ticker ASC
                """,
                conn,
                params=(asof,),
            )
        raw = pd.read_sql_query(
            """
            SELECT commodity, ticker, published_at, source, url, headline, sentiment, event_tags
            FROM commodity_news_raw
            ORDER BY published_at DESC, fetched_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    return {
        "asof_date": asof,
        "features": features.to_dict(orient="records") if not features.empty else [],
        "articles": raw.to_dict(orient="records") if not raw.empty else [],
    }


@app.get("/commodities/trades/latest")
def commodities_trades_latest():
    with get_conn() as conn:
        asof = _latest_asof_on_or_before_prices(conn, "commodity_model_trades")
        if asof is None:
            return {"asof_date": None, "trades": []}
        df = pd.read_sql_query(
            """
            SELECT asof_date, ticker, trade_action, shares_delta, est_notional
            FROM commodity_model_trades
            WHERE asof_date = ?
            ORDER BY est_notional DESC, ticker ASC
            """,
            conn,
            params=(asof,),
        )
    return {"asof_date": asof, "trades": df.to_dict(orient="records")}


@app.get("/commodities/export/trading212.csv")
def commodities_export_trading212_csv():
    with get_conn() as conn:
        asof = _latest_asof_on_or_before_prices(conn, "commodity_model_holdings")
        if asof is None:
            return Response(content="message\nNo commodity recommendations found\n", media_type="text/csv")
        df = _commodity_holdings_frame(conn, asof)
    noncash = df[df["ticker"].astype(str).str.lower() != "cash"].copy()
    if noncash.empty:
        return Response(content="message\nNo invested commodity holdings found\n", media_type="text/csv")
    total = float(noncash["invested_weight"].astype(float).sum())
    if total > 0:
        noncash["invested_weight"] = noncash["invested_weight"].astype(float) / total
    export = pd.DataFrame(
        {
            "asof_date": asof,
            "project_ticker": noncash["ticker"],
            "trading212_ticker": noncash["trading212_ticker"],
            "trading212_search_name": noncash["trading212_name"],
            "commodity": noncash["commodity"],
            "pie_weight": noncash["invested_weight"].astype(float).round(6),
            "pie_percent": (noncash["invested_weight"].astype(float) * 100.0).round(2),
            "target_weight_including_cash": noncash["target_weight"].astype(float).round(6),
            "reason": noncash["reasons"],
        }
    )
    buf = io.StringIO()
    export.to_csv(buf, index=False)
    filename = f"commodity_scout_trading212_{asof}.csv"
    return Response(content=buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/commodities/backtest")
def commodities_backtest(cost_bps: float = 5.0, max_points: int = 2000, start_date: str = "2015-01-01"):
    curve, stats = run_commodity_backtest(cost_bps=cost_bps, start_date=start_date)
    if curve.empty:
        return {"stats": stats, "series": [], "benchmark": []}
    stats["end_date"] = str(pd.to_datetime(curve["date"]).max().date())
    stats["note"] = (
        "Default view starts in 2015 because the older commodity/futures history is sparse, "
        "less comparable to the current Trading 212 ETC-style pie, and can make the chart look "
        "like a forecast when it is only an old stress-test."
    )
    if len(curve) > max_points:
        curve = curve.iloc[:: max(1, len(curve) // max_points)].copy()

    benchmark = []
    with get_conn() as conn:
        dbc = pd.read_sql_query(
            """
            SELECT date, close
            FROM prices_daily
            WHERE ticker = 'dbc.us'
            ORDER BY date ASC
            """,
            conn,
        )
    if not dbc.empty:
        dbc["date"] = pd.to_datetime(dbc["date"])
        start = pd.to_datetime(curve["date"]).min()
        end = pd.to_datetime(curve["date"]).max()
        dbc = dbc[(dbc["date"] >= start) & (dbc["date"] <= end)].copy()
        if not dbc.empty:
            base = float(dbc["close"].iloc[0])
            dbc["equity"] = dbc["close"].astype(float) / base
            if len(dbc) > max_points:
                dbc = dbc.iloc[:: max(1, len(dbc) // max_points)].copy()
            benchmark = [{"date": str(d.date()), "equity": float(e)} for d, e in zip(dbc["date"], dbc["equity"])]

    series = [
        {"date": str(d.date()), "equity": float(e)}
        for d, e in zip(pd.to_datetime(curve["date"]), curve["equity"])
    ]
    return {"stats": stats, "series": series, "benchmark": benchmark, "benchmark_name": "DBC broad commodities"}


def _empty_portfolio_performance(asof_date=None):
    return {
        "asof_date": asof_date,
        "end_date": None,
        "start_equity": None,
        "end_equity": None,
        "current_equity": None,
        "current_snapshot_date": None,
        "pnl_gbp": None,
        "pnl_pct": None,
        "mom_change_gbp": None,
        "mom_change_pct": None,
        "max_drawdown": None,
        "spy_start": None,
        "spy_end": None,
        "spy_pnl_pct": None,
        "contributors": [],
        "best_contributors": [],
        "worst_contributors": [],
        "curve": [],
    }


@app.get("/portfolio/history")
def portfolio_history(limit: int = Query(60, ge=1, le=240)):
    return {"history": get_portfolio_history(limit=limit)}


@app.get("/portfolio/performance/latest")
def portfolio_performance_latest():
    data = get_portfolio_performance()
    return data if data is not None else _empty_portfolio_performance()


@app.get("/portfolio/performance")
def portfolio_performance(asof_date: str = Query(...)):
    data = get_portfolio_performance(asof_date=asof_date)
    return data if data is not None else _empty_portfolio_performance(asof_date=asof_date)


def _load_prices_and_recs():
    universe = get_universe()
    placeholders = ",".join(["?"] * len(universe))
    with get_conn() as conn:
        prices = pd.read_sql_query(
            f"""
            SELECT date, ticker, close
            FROM prices_daily
            WHERE ticker IN ({placeholders}) AND date >= ?
            ORDER BY date ASC
            """,
            conn,
            params=[*universe, BACKTEST_START_DATE],
        )
        recs = pd.read_sql_query(
            """
            SELECT asof_date, ticker, target_weight
            FROM recommendations
            WHERE asof_date >= ?
            ORDER BY asof_date ASC
            """,
            conn,
            params=(BACKTEST_START_DATE,),
        )
    return prices, recs


@app.get("/backtest/equity")
def backtest_equity(cost_bps: float = 5.0, max_points: int = 2000):
    prices, recs = _load_prices_and_recs()
    curve, stats, benchmark_6040 = run_recommendation_backtest(prices, recs, cost_bps=cost_bps)
    if curve.empty:
        return {"stats": stats, "series": [], "benchmark_6040": []}
    if len(curve) > max_points:
        curve = curve.iloc[:: max(1, len(curve) // max_points)].copy()
    series = [
        {"date": str(d.date()), "equity": float(e)}
        for d, e in zip(pd.to_datetime(curve["date"]), curve["equity"])
    ]
    if len(benchmark_6040) > max_points:
        benchmark_6040 = benchmark_6040.iloc[:: max(1, len(benchmark_6040) // max_points)].copy()
    benchmark_series = [
        {"date": str(pd.Timestamp(d).date()), "equity": float(e)}
        for d, e in benchmark_6040.items()
    ]
    return {"stats": stats, "series": series, "benchmark_6040": benchmark_series}


@app.get("/export/rebalance_pack.csv")
def export_rebalance_pack_csv(portfolio_value: float | None = Query(None, gt=0)):
    try:
        effective_portfolio_value = float(portfolio_value) if portfolio_value is not None else _current_portfolio_value()
    except (TypeError, ValueError):
        effective_portfolio_value = _current_portfolio_value()
    with get_conn() as conn:
        dates = _recent_asof_dates(conn, "model_holdings", 2)
        if dates.empty:
            dates = _recent_recommendation_dates(conn, 2)
        if dates.empty:
            return Response(content="message\nNo recommendations found\n", media_type="text/csv")

        latest_asof = dates["asof_date"].iloc[0]
        prev_asof = dates["asof_date"].iloc[1] if len(dates) > 1 else None

        latest = pd.read_sql_query(
            """
            SELECT
                h.ticker,
                h.target_weight,
                COALESCE(r.action, CASE WHEN h.ticker = 'cash' THEN 'HOLD_CASH' ELSE 'BUY_OR_HOLD' END) AS action,
                COALESCE(r.score, 0.0) AS score,
                COALESCE(r.reasons, '') AS reasons
            FROM model_holdings h
            LEFT JOIN recommendations r
              ON r.asof_date = h.asof_date
             AND r.ticker = h.ticker
            WHERE h.asof_date = ? AND h.target_weight > 0
            """,
            conn,
            params=(latest_asof,),
        )
        prev = pd.DataFrame(columns=["ticker", "target_weight"])
        if prev_asof:
            prev = pd.read_sql_query(
                "SELECT ticker, target_weight FROM model_holdings WHERE asof_date = ? AND target_weight > 0",
                conn,
                params=(prev_asof,),
            )
        prices = pd.read_sql_query(
            """
            SELECT p1.ticker, p1.close AS price FROM prices_daily p1
            JOIN (SELECT ticker, MAX(date) AS max_date FROM prices_daily GROUP BY ticker) p2
            ON p1.ticker = p2.ticker AND p1.date = p2.max_date
            """,
            conn,
        )
        prices = pd.concat(
            [prices, pd.DataFrame([{"ticker": CASH_TICKER, "price": 1.0}])],
            ignore_index=True,
        )

    latest_w = latest[["ticker", "target_weight"]].rename(columns={"target_weight": "new_weight"})
    prev_w = prev[["ticker", "target_weight"]].rename(columns={"target_weight": "prev_weight"}) if not prev.empty else pd.DataFrame(columns=["ticker", "prev_weight"])

    pack = latest_w.merge(prev_w, on="ticker", how="outer").fillna(0.0)
    pack = pack.merge(latest[["ticker", "action", "score", "reasons"]], on="ticker", how="left")
    pack = pack.merge(prices, on="ticker", how="left")
    pack["delta_weight"] = pack["new_weight"] - pack["prev_weight"]

    def classify(row):
        if str(row["ticker"]).lower() == CASH_TICKER:
            return "CASH_RESERVE"
        nw, pw = float(row["new_weight"]), float(row["prev_weight"])
        if pw == 0 and nw > 0: return "NEW_BUY"
        if pw > 0 and nw == 0: return "EXIT_SELL"
        if nw > pw: return "BUY"
        if nw < pw: return "SELL"
        return "HOLD"

    pack["rebalance_action"] = pack.apply(classify, axis=1)
    pack["portfolio_value"] = round(effective_portfolio_value, 2)
    def estimate_shares_delta(row):
        if str(row["ticker"]).lower() == CASH_TICKER:
            return 0.0
        if pd.isna(row["price"]) or float(row["price"]) == 0.0:
            return 0.0
        return (float(row["delta_weight"]) * effective_portfolio_value) / float(row["price"])

    pack["est_shares_delta"] = pack.apply(estimate_shares_delta, axis=1)
    pack["est_notional"] = pack["est_shares_delta"].abs() * pack["price"].fillna(0.0)

    action_order = {"NEW_BUY": 0, "BUY": 1, "SELL": 2, "EXIT_SELL": 3, "HOLD": 4, "CASH_RESERVE": 5}
    pack["sort_key"] = pack["rebalance_action"].map(action_order).fillna(9)
    pack = pack.sort_values(["sort_key", "est_notional"], ascending=[True, False]).drop(columns=["sort_key"])

    for c in ["prev_weight", "new_weight", "delta_weight"]:
        pack[c] = pack[c].astype(float).round(6)
    pack["price"] = pack["price"].astype(float).round(4)
    pack["est_shares_delta"] = pack["est_shares_delta"].astype(float).round(6)
    pack["est_notional"] = pack["est_notional"].astype(float).round(2)

    pack.insert(0, "latest_asof_date", latest_asof)
    pack.insert(1, "previous_asof_date", prev_asof if prev_asof else "")

    cols = ["latest_asof_date", "previous_asof_date", "ticker", "prev_weight", "new_weight", "delta_weight", "rebalance_action", "price", "portfolio_value", "est_shares_delta", "est_notional", "score", "action", "reasons"]
    for c in cols:
        if c not in pack.columns:
            pack[c] = ""
    pack = pack[cols]

    buf = io.StringIO()
    pack.to_csv(buf, index=False)
    filename = f"rebalance_pack_{latest_asof}.csv"
    return Response(content=buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ═══════════════════════════════════════════════════════════
# Dashboard
# ═══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Investing Dashboard</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0b0f16; color: #e7eefc; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }
    h1 { margin: 0; font-size: 22px; }
    h2 { margin: 0 0 10px; font-size: 16px; color: #d9e4ff; font-weight: 700; }
    .sub { color: #8b98b0; font-size: 13px; margin-top: 4px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 20px; }
    @media (min-width: 1000px) { .grid { grid-template-columns: 1.3fr 0.7fr; } }
    .card { background: #101826; border: 1px solid #1b2740; border-radius: 14px; padding: 16px; }
    .pill { display: inline-block; background: #0d1420; border: 1px solid #1b2740; border-radius: 999px; padding: 6px 12px; font-size: 12px; color: #cfe0ff; }
    .pill b { color: #fff; font-weight: 700; }
    .muted { color: #8b98b0; }
    .small { font-size: 12px; }
    .footer { margin-top: 16px; color: #6b7a94; font-size: 12px; }
    a { color: #93b4ff; text-decoration: none; }
    input, button { background: #0d1420; border: 1px solid #1b2740; color: #e7eefc; border-radius: 10px; padding: 8px 12px; font-size: 13px; }
    input { width: 100px; }
    button { cursor: pointer; }
    button:hover { background: #1a2a4a; }
    canvas { width: 100% !important; height: 300px !important; }

    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
    .tab-nav { display: inline-flex; gap: 6px; margin-top: 18px; background: #0d1420; border: 1px solid #1b2740; border-radius: 999px; padding: 5px; }
    .tab-button { border: 0; border-radius: 999px; padding: 8px 14px; color: #8b98b0; background: transparent; font-weight: 700; }
    .tab-button.active { background: #1e3a5f; color: #dbeafe; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .tf-btns { display: flex; gap: 4px; }
    .tf-btn { padding: 5px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; border: 1px solid #1b2740; background: #0d1420; color: #8b98b0; cursor: pointer; }
    .tf-btn.active { background: #1e3a5f; color: #93b4ff; border-color: #2d5a8e; }

    .market-box { background: #0d1420; border: 1px solid #1b2740; border-radius: 10px; padding: 12px 14px; margin-top: 12px; font-size: 13px; line-height: 1.65; color: #b0bdd4; }
    .market-box b { color: #e7eefc; }
    .market-box .warn { color: #fbbf24; }
    .market-box .ok { color: #86efac; }

    .holding-card { background: #0d1420; border: 1px solid #1b2740; border-radius: 12px; padding: 12px 14px; margin-bottom: 8px; }
    .hc-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
    .hc-name { font-weight: 700; font-size: 14px; color: #e7eefc; }
    .hc-full { font-size: 12px; color: #8b98b0; font-weight: 400; }
    .hc-weight { font-size: 13px; color: #93b4ff; font-weight: 600; white-space: nowrap; }
    .hc-bar { height: 4px; border-radius: 4px; margin: 8px 0 6px; background: #1b2740; }
    .hc-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .hc-details { font-size: 12px; color: #8b98b0; line-height: 1.5; }
    .hc-reason { font-size: 13px; color: #b0bdd4; margin-top: 6px; line-height: 1.6; }
    .hc-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .mini-pill { display: inline-flex; align-items: center; gap: 4px; background: #111d30; border: 1px solid #223452; border-radius: 999px; color: #aebbd2; font-size: 11px; padding: 4px 8px; }
    .cash-card { border-color: rgba(96,165,250,0.38); background: linear-gradient(135deg, rgba(30,58,95,0.36), #0d1420 60%); }
    .allocation-check { margin-top: 12px; padding: 10px 12px; border-radius: 12px; background: #0d1420; border: 1px solid #1b2740; display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .allocation-check.ok { border-color: rgba(74,222,128,0.25); }
    .allocation-check.warn { border-color: rgba(251,191,36,0.45); }
    .allocation-breakdown { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
    .allocation-chip { background: #0d1420; border: 1px solid #1b2740; border-radius: 10px; padding: 8px 10px; font-size: 12px; color: #aebbd2; }
    .allocation-chip b { display: block; color: #f7fbff; font-size: 14px; margin-top: 2px; }

    .change-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 10px; margin-bottom: 6px; font-size: 13px; }
    .change-row.buy { background: rgba(74,222,128,0.06); border: 1px solid rgba(74,222,128,0.15); }
    .change-row.sell { background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.15); }
    .change-row.new { background: rgba(96,165,250,0.08); border: 1px solid rgba(96,165,250,0.2); }
    .change-row.exit { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.2); }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; min-width: 55px; text-align: center; }
    .badge.new { background: #1e3a5f; color: #93c5fd; }
    .badge.buy { background: #14532d; color: #86efac; }
    .badge.sell { background: #7f1d1d; color: #fca5a5; }
    .badge.exit { background: #7f1d1d; color: #fca5a5; }

    .action-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-radius: 10px; margin-bottom: 6px; font-size: 13px; background: #0d1420; border: 1px solid #1b2740; }
    .ar-left { display: flex; align-items: center; gap: 10px; }
    .ar-right { text-align: right; color: #8b98b0; font-size: 12px; }
    .btnrow { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .month-select { min-width: 210px; width: auto; }
    .metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 12px 0; }
    @media (min-width: 700px) { .metric-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
    .metric-card { background: #0d1420; border: 1px solid #1b2740; border-radius: 12px; padding: 12px; }
    .metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #7f8aa3; }
    .metric-value { font-size: 20px; font-weight: 700; color: #f7fbff; margin-top: 6px; }
    .metric-sub { font-size: 12px; color: #8b98b0; margin-top: 4px; }
    .metric-value.pos { color: #86efac; }
    .metric-value.neg { color: #fca5a5; }
    .perf-meta { color: #b0bdd4; font-size: 13px; line-height: 1.65; margin-top: 8px; }
    .contrib-grid { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 14px; }
    @media (min-width: 700px) { .contrib-grid { grid-template-columns: 1fr 1fr; } }
    .contrib-list { display: grid; gap: 8px; }
    .contrib-item { background: #0d1420; border: 1px solid #1b2740; border-radius: 10px; padding: 10px 12px; display: flex; justify-content: space-between; gap: 12px; }
    .contrib-left { display: flex; flex-direction: column; gap: 2px; }
    .contrib-right { text-align: right; font-size: 12px; color: #8b98b0; }
    .chart-wrap { margin-top: 14px; }
    #monthCurveChart { width: 100% !important; height: 250px !important; }
    .commodity-hero { background: radial-gradient(circle at 18% 0%, rgba(245,158,11,0.20), transparent 34%), linear-gradient(135deg, #121b2b, #0d1420); border: 1px solid #263551; border-radius: 16px; padding: 16px; margin-top: 20px; }
    .commodity-hero h2 { font-size: 20px; }
    .commodity-grid { display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 16px; }
    @media (min-width: 1000px) { .commodity-grid { grid-template-columns: 1fr 1fr; } }
    .commodity-card { background: #0d1420; border: 1px solid #1b2740; border-radius: 12px; padding: 12px 14px; margin-bottom: 8px; }
    .commodity-card.cash { border-color: rgba(245,158,11,0.38); background: linear-gradient(135deg, rgba(245,158,11,0.10), #0d1420 62%); }
    .commodity-top { display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }
    .commodity-weight { color: #fbbf24; font-weight: 800; }
    .news-row { padding: 10px 12px; background: #0d1420; border: 1px solid #1b2740; border-radius: 10px; margin-bottom: 8px; font-size: 12px; color: #b0bdd4; line-height: 1.45; }
    .pie-chip-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 8px; margin-top: 10px; }
    .pie-chip { border: 1px solid #24344e; background: #101826; border-radius: 12px; padding: 10px; font-size: 12px; color: #aebbd2; }
    .pie-chip b { display: block; color: #fff; font-size: 16px; margin-top: 2px; }
    #commodityChart { width: 100% !important; height: 240px !important; }
  </style>
</head>
<body>
<div class="wrap">
  <div style="display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:8px;">
    <div>
      <h1>Portfolio Dashboard</h1>
      <div class="sub">ML-powered monthly portfolio with volatility targeting and drawdown protection.</div>
    </div>
    <div class="controls">
      <span class="pill">Trade Pack <b id="pvLabel">__DEFAULT_PORTFOLIO_VALUE__</b></span>
      <input id="portfolioValue" type="number" value="__DEFAULT_PORTFOLIO_VALUE__" min="1" step="50">
      <button onclick="reloadAll()">Refresh</button>
    </div>
  </div>

  <div class="tab-nav">
    <button class="tab-button active" onclick="setDashboardTab('everything')">Everything</button>
    <button class="tab-button" onclick="setDashboardTab('commodities')">Commodities Only</button>
  </div>

  <div id="everythingTab" class="tab-panel active">
  <div class="grid">
    <!-- ═══ LEFT COLUMN ═══ -->
    <div>
      <!-- Backtest -->
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:10px;">
          <h2 style="margin:0;">Portfolio Performance</h2>
          <div class="tf-btns">
            <div class="tf-btn" onclick="setTimeframe('1Y')">1Y</div>
            <div class="tf-btn" onclick="setTimeframe('2Y')">2Y</div>
            <div class="tf-btn" onclick="setTimeframe('5Y')">5Y</div>
            <div class="tf-btn active" onclick="setTimeframe('ALL')">All</div>
          </div>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;" id="btStats"></div>
        <canvas id="equityChart"></canvas>
      </div>

      <div class="card" style="margin-top:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <h2 style="margin:0;">Actual Portfolio</h2>
          <select id="monthPicker" class="month-select"></select>
        </div>
        <div id="perfMeta" class="perf-meta">Loading...</div>
        <div id="perfMetrics" class="metric-grid"></div>
        <div class="chart-wrap">
          <canvas id="monthCurveChart"></canvas>
        </div>
        <div class="contrib-grid">
          <div>
            <p class="muted small">Best contributors</p>
            <div id="bestContrib" class="contrib-list"></div>
          </div>
          <div>
            <p class="muted small">Worst contributors</p>
            <div id="worstContrib" class="contrib-list"></div>
          </div>
        </div>
      </div>

      <!-- Market conditions -->
      <div class="card" style="margin-top:16px;">
        <h2>Market Conditions Right Now</h2>
        <div id="marketBox" class="market-box">Loading...</div>
      </div>
    </div>

    <!-- ═══ RIGHT COLUMN ═══ -->
    <div>
      <!-- Summary -->
      <div class="card">
        <h2>This Month's Portfolio</h2>
        <div id="summaryText" style="font-size:13.5px; color:#b0bdd4; line-height:1.7;">Loading...</div>
      </div>

      <!-- Holdings -->
      <div class="card" style="margin-top:16px; max-height:60vh; overflow-y:auto;">
        <h2>Your Holdings</h2>
        <p class="muted small" id="holdingsExplainer"></p>
        <div id="holdingsCards"></div>
      </div>

      <!-- Changes -->
      <div class="card" style="margin-top:16px;">
        <h2>What Changed</h2>
        <p class="muted small" id="changesExplainer"></p>
        <div id="changesList"></div>
      </div>

      <!-- Actions -->
      <div class="card" style="margin-top:16px;">
        <h2>Trades to Make</h2>
        <p class="muted small">The actual trades to execute in Trading 212.</p>
        <div id="actionsList"></div>
        <div class="btnrow">
          <button onclick="downloadPieCSV()">Download Allocation CSV</button>
          <button onclick="downloadRebalancePack()">Download Full CSV</button>
        </div>
      </div>
    </div>
  </div>
  </div>

  <div id="commoditiesTab" class="tab-panel">
    <div class="commodity-hero">
      <h2>Commodity Scout</h2>
      <div class="sub">A separate commodities-only Trading 212 pie. Cash is shown outside the pie; invested weights are normalized to 100% for Trading 212.</div>
      <div id="commoditySummary" style="margin-top:12px; font-size:13.5px; color:#b0bdd4; line-height:1.7;">Loading...</div>
    </div>
    <div class="commodity-grid">
      <div>
        <div class="card">
          <h2>Commodity Pie Performance</h2>
          <div id="commodityStats" style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;"></div>
          <canvas id="commodityChart"></canvas>
        </div>
        <div class="card" style="margin-top:16px;">
          <h2>News Drivers</h2>
          <p class="muted small">Google News RSS articles grouped by commodity theme. If the feed fails, Commodity Scout falls back to price and macro signals.</p>
          <div id="commodityNews"></div>
        </div>
      </div>
      <div>
        <div class="card">
          <h2>Commodity Holdings</h2>
          <p class="muted small" id="commodityHoldingsExplainer"></p>
          <div id="commodityHoldings"></div>
        </div>
        <div class="card" style="margin-top:16px;">
          <h2>Trading 212 Pie Weights</h2>
          <p class="muted small">Use these percentages inside the commodities pie. Cash is not entered into the pie.</p>
          <div id="commodityPieWeights"></div>
          <div class="btnrow"><button onclick="downloadCommodityCSV()">Download Commodity Pie CSV</button></div>
        </div>
        <div class="card" style="margin-top:16px;">
          <h2>Commodity Trades</h2>
          <p class="muted small">Actionable buy/sell deltas only. Cash changes are reflected through buys/sells, not as a fake stock.</p>
          <div id="commodityTrades"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">Run <code>python run_pipeline.py</code> for Everything and <code>python run_commodity_pipeline.py</code> for Commodity Scout. Keep <code>python run_api.py --serve</code> running for this UI.</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<script>
// ═══════════════════════════════════════════════════════
// Ticker → friendly name map
// ═══════════════════════════════════════════════════════
const NAMES = {
  "cash":"Cash / Dry Powder",
  // ETFs
  "spy.us":"S&P 500 Index","qqq.us":"Nasdaq 100 Index","dia.us":"Dow Jones Index","iwm.us":"Russell 2000 Small Cap",
  "vti.us":"Total US Stock Market","voo.us":"S&P 500 (Vanguard)","ief.us":"7-10 Year US Treasuries","tlt.us":"20+ Year US Treasuries",
  "shy.us":"Short-Term Treasuries","lqd.us":"Investment Grade Bonds","hyg.us":"High Yield (Junk) Bonds",
  "gld.us":"Gold","iau.us":"Gold (iShares)","slv.us":"Silver",
  "dbc.us":"Broad Commodities","pdbc.us":"Broad Commodities (Invesco)","djp.us":"Commodity Index",
  "uso.us":"Oil","ung.us":"Natural Gas","dba.us":"Agriculture","cper.us":"Copper","copx.us":"Copper Miners",
  "cl=f":"WTI Crude Oil Futures","bz=f":"Brent Crude Oil Futures","ho=f":"Heating Oil / Jet Fuel Proxy",
  "rb=f":"Gasoline Futures","ng=f":"Natural Gas Futures","gc=f":"Gold Futures","si=f":"Silver Futures",
  "hg=f":"Copper Futures","pl=f":"Platinum Futures","pa=f":"Palladium Futures","zc=f":"Corn Futures",
  "zw=f":"Wheat Futures","zs=f":"Soybean Futures","kc=f":"Coffee Futures","cc=f":"Cocoa Futures",
  "ct=f":"Cotton Futures","sb=f":"Sugar Futures","le=f":"Live Cattle Futures","he=f":"Lean Hogs Futures",
  "oj=f":"Orange Juice Futures",
  // Big tech
  "aapl.us":"Apple","msft.us":"Microsoft","googl.us":"Alphabet (Google)","amzn.us":"Amazon","meta.us":"Meta (Facebook)",
  "nvda.us":"Nvidia","tsla.us":"Tesla","nflx.us":"Netflix","crm.us":"Salesforce","adbe.us":"Adobe",
  "orcl.us":"Oracle","csco.us":"Cisco","intc.us":"Intel","ibm.us":"IBM",
  // Semis
  "amd.us":"AMD","avgo.us":"Broadcom","qcom.us":"Qualcomm","txn.us":"Texas Instruments",
  "amat.us":"Applied Materials","lrcx.us":"Lam Research","klac.us":"KLA Corp","mrvl.us":"Marvell",
  "nxpi.us":"NXP Semiconductors","mu.us":"Micron","mchp.us":"Microchip Tech","arm.us":"ARM Holdings",
  "asml.us":"ASML","smci.us":"Super Micro Computer","on.us":"ON Semi","adi.us":"Analog Devices",
  // Healthcare
  "unh.us":"UnitedHealth","jnj.us":"Johnson & Johnson","lly.us":"Eli Lilly","pfe.us":"Pfizer",
  "abbv.us":"AbbVie","mrk.us":"Merck","tmo.us":"Thermo Fisher","abt.us":"Abbott Labs",
  "amgn.us":"Amgen","gild.us":"Gilead Sciences","isrg.us":"Intuitive Surgical","vrtx.us":"Vertex Pharma",
  "regn.us":"Regeneron","hca.us":"HCA Healthcare","bmy.us":"Bristol-Myers","ci.us":"Cigna",
  "hum.us":"Humana","elv.us":"Elevance Health","syk.us":"Stryker","bsx.us":"Boston Scientific",
  // Financials
  "jpm.us":"JPMorgan Chase","bac.us":"Bank of America","wfc.us":"Wells Fargo","gs.us":"Goldman Sachs",
  "ms.us":"Morgan Stanley","c.us":"Citigroup","blk.us":"BlackRock","schw.us":"Charles Schwab",
  "spgi.us":"S&P Global","v.us":"Visa","ma.us":"Mastercard","axp.us":"American Express","pypl.us":"PayPal",
  // Industrial
  "cat.us":"Caterpillar","de.us":"John Deere","hon.us":"Honeywell","ge.us":"GE Aerospace",
  "ba.us":"Boeing","rtx.us":"RTX (Raytheon)","lmt.us":"Lockheed Martin","ups.us":"UPS","fdx.us":"FedEx",
  // Consumer
  "wmt.us":"Walmart","cost.us":"Costco","hd.us":"Home Depot","low.us":"Lowe's","tgt.us":"Target",
  "ko.us":"Coca-Cola","pep.us":"PepsiCo","pg.us":"Procter & Gamble","mcd.us":"McDonald's",
  "sbux.us":"Starbucks","nke.us":"Nike","cl.us":"Colgate-Palmolive",
  // Energy
  "xom.us":"ExxonMobil","cvx.us":"Chevron","cop.us":"ConocoPhillips","slb.us":"Schlumberger",
  "oxy.us":"Occidental Petroleum","vlo.us":"Valero Energy","mpc.us":"Marathon Petroleum",
  // Utilities/REITs
  "nee.us":"NextEra Energy","duk.us":"Duke Energy","so.us":"Southern Company",
  "pld.us":"Prologis REIT","amt.us":"American Tower REIT","o.us":"Realty Income REIT",
  // Tech/Software
  "now.us":"ServiceNow","uber.us":"Uber","abnb.us":"Airbnb","shop.us":"Shopify",
  "pltr.us":"Palantir","snow.us":"Snowflake","crwd.us":"CrowdStrike","panw.us":"Palo Alto Networks",
  "net.us":"Cloudflare","ddog.us":"Datadog","ftnt.us":"Fortinet","zs.us":"Zscaler",
  "sq.us":"Block (Square)","snap.us":"Snap","pins.us":"Pinterest",
  // Storage
  "wdc.us":"Western Digital","stt.us":"State Street",
  // Other
  "brk.b.us":"Berkshire Hathaway",
};

function getName(t) {
  const n = NAMES[t.toLowerCase()];
  if (n) return n;
  return t.replace(/\.us$/i,"").toUpperCase();
}
function getTicker(t) {
  t = String(t || "").toLowerCase();
  if (t === "cash") return "CASH";
  return t.replace(/\.us$/i,"").toUpperCase();
}

function bucketLabel(t) {
  t = t.toLowerCase();
  if (t === "cash") return "Cash";
  const bonds = ["ief.us","tlt.us","shy.us","lqd.us","hyg.us"];
  const comms = ["gld.us","iau.us","slv.us","dbc.us","pdbc.us","djp.us","uso.us","ung.us","dba.us","cper.us","copx.us","cl=f","bz=f","ho=f","rb=f","ng=f","gc=f","si=f","hg=f","pl=f","pa=f","zc=f","zw=f","zs=f","kc=f","cc=f","ct=f","sb=f","le=f","he=f","oj=f"];
  if (bonds.includes(t)) return "Bond ETF";
  if (comms.includes(t)) return "Commodity";
  const etfs = ["spy.us","qqq.us","dia.us","iwm.us","vti.us","voo.us"];
  if (etfs.includes(t)) return "Index ETF";
  return "Stock";
}

// ═══════════════════════════════════════════════════════
// Globals
// ═══════════════════════════════════════════════════════
let equityChart = null, monthCurveChart = null, commodityChart = null, latestRecs = null, prevRecs = null, latestPrices = null, latestTrades = null;
let portfolioHistory = [], latestPerformance = null, selectedPerformance = null, selectedPerformanceAsOf = null;
let allSeries = [], currentTimeframe = "ALL";
let commodityLatest = null, commodityNews = null, commodityTrades = null, commodityBacktest = null;

const fmtPct = x => x == null ? "n/a" : (x*100).toFixed(1)+"%";
const fmtGBP = x => x == null ? "n/a" : "\u00a3"+Number(x).toFixed(2);
const fmtSignedGBP = x => x == null ? "n/a" : `${Number(x) >= 0 ? "+" : "-"}\u00a3${Math.abs(Number(x)).toFixed(2)}`;
const fmtNum = (x,d=2) => x == null ? "n/a" : Number(x).toFixed(d);
const csvCell = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
const DEFAULT_PORTFOLIO_VALUE = __DEFAULT_PORTFOLIO_VALUE__;
async function fetchJson(u) { return (await fetch(u)).json(); }
function priceMap() { const m={cash:1}; (latestPrices||[]).forEach(p=>m[p.ticker]=Number(p.close)); return m; }
function getPV() { const v=Number(document.getElementById("portfolioValue").value||0); return v>0?v:DEFAULT_PORTFOLIO_VALUE; }
function downloadRebalancePack() { window.location="/export/rebalance_pack.csv?portfolio_value="+getPV(); }
function perfClass(x) { return x == null ? "" : (Number(x) > 0 ? "pos" : (Number(x) < 0 ? "neg" : "")); }

function setDashboardTab(tab) {
  document.getElementById("everythingTab").classList.toggle("active", tab === "everything");
  document.getElementById("commoditiesTab").classList.toggle("active", tab === "commodities");
  document.querySelectorAll(".tab-button").forEach(btn => {
    const isComms = btn.textContent.toLowerCase().includes("commodities");
    btn.classList.toggle("active", tab === (isComms ? "commodities" : "everything"));
  });
  if (tab === "commodities") renderCommodityAll();
}

// ═══════════════════════════════════════════════════════
// Reason humaniser
// ═══════════════════════════════════════════════════════
function humaniseReasons(ticker, reasons, weight) {
  if (!reasons) return "";
  const t = ticker.toLowerCase();
  const name = getName(t);
  const bucket = bucketLabel(t);
  let parts = [];

  // Why selected
  if (bucket === "Cash") {
    parts.push(`Cash is being held as dry powder. It keeps part of the portfolio stable and gives the model room to buy more risk assets when conditions improve.`);
  } else if (bucket === "Bond ETF") {
    parts.push(`${name} is included to add stability. Bonds tend to hold their value or rise when stocks fall, acting as a cushion for the portfolio.`);
  } else if (bucket === "Commodity") {
    parts.push(`${name} is included for diversification. Commodities like gold or silver often move independently from stocks, which helps protect against inflation and market shocks.`);
  } else {
    parts.push(`${name} was picked because the model thinks it's likely to perform well relative to other stocks over the next month.`);
  }

  // Momentum context
  if (reasons.includes("12m momentum:")) {
    const m = reasons.match(/12m momentum: ([\-\d.]+)/);
    if (m) {
      const v = parseFloat(m[1]);
      if (v > 0.3) parts.push(`Its price has risen ${(v*100).toFixed(0)}% over the past year, showing strong upward momentum.`);
      else if (v > 0.1) parts.push(`It's been trending upward over the past year, gaining about ${(v*100).toFixed(0)}%.`);
      else if (v > 0) parts.push(`It's been slightly positive over the past year.`);
      else if (v > -0.1) parts.push(`It's been roughly flat over the past year.`);
      else parts.push(`Its price has fallen ${(Math.abs(v)*100).toFixed(0)}% over the past year, but the model sees recovery potential.`);
    }
  }

  // Trend
  if (reasons.includes("Above 200d MA")) parts.push("It's trading above its long-term average price, which is generally a healthy sign.");
  if (reasons.includes("Below 200d MA")) parts.push("It's currently below its long-term average, which can mean it's undervalued or the model sees a turnaround coming.");

  // Volume
  if (reasons.includes("Unusually high recent trading volume")) parts.push("There's been unusually high trading activity recently, which often signals that something significant is happening.");

  // Macro
  if (reasons.includes("Elevated market volatility")) parts.push("The overall market is quite volatile right now, so the model has been cautious with sizing.");

  // DD tilt
  if (reasons.includes("DD tilt")) parts.push("The model has shifted some money from stocks to bonds because the market has been falling recently.");

  if (reasons.includes("Performance feedback:")) {
    const m = reasons.match(/Performance feedback: ([+\-\d.]+)/);
    if (m) {
      const v = parseFloat(m[1]);
      if (v > 0) parts.push("Recent real portfolio performance is also nudging this name higher.");
      if (v < 0) parts.push("Recent real portfolio performance is putting pressure on this name's size.");
    }
  }

  return parts.join(" ");
}

// ═══════════════════════════════════════════════════════
// Backtest chart with timeframes
// ═══════════════════════════════════════════════════════
function renderStats(stats) {
  document.getElementById("btStats").innerHTML = `
    <span class="pill">Annual return <b>${fmtPct(stats.cagr)}</b></span>
    <span class="pill">Risk <b>${fmtPct(stats.vol)}</b></span>
    <span class="pill">Sharpe <b>${stats.sharpe!=null?stats.sharpe.toFixed(2):"n/a"}</b></span>
    <span class="pill">Worst drop <b>${fmtPct(stats.max_drawdown)}</b></span>
  `;
}

function filterSeries(tf) {
  if (!allSeries.length) return allSeries;
  const last = new Date(allSeries[allSeries.length-1].date);
  let cutoff = new Date(0);
  if (tf === "1Y") cutoff = new Date(last.getFullYear()-1, last.getMonth(), last.getDate());
  else if (tf === "2Y") cutoff = new Date(last.getFullYear()-2, last.getMonth(), last.getDate());
  else if (tf === "5Y") cutoff = new Date(last.getFullYear()-5, last.getMonth(), last.getDate());
  return allSeries.filter(p => new Date(p.date) >= cutoff);
}

function buildChart(series) {
  const ctx = document.getElementById("equityChart");
  if (equityChart) equityChart.destroy();

  // Rebase to 100 for readability
  const base = series.length ? series[0].equity : 1;
  const data = series.map(p => ({ x: p.date, y: (p.equity / base) * 100 }));

  equityChart = new Chart(ctx, {
    type: "line",
    data: { datasets: [{ data, borderColor: "#3b82f6", borderWidth: 2, pointRadius: 0, tension: 0.1, fill: { target: "origin", above: "rgba(59,130,246,0.08)" } }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => " Value: " + c.parsed.y.toFixed(1) } } },
      scales: {
        x: { type: "time", time: { unit: series.length > 500 ? "quarter" : "month" }, grid: { color: "rgba(140,180,255,0.06)" }, ticks: { color: "#6b7a94", maxTicksLimit: 8 } },
        y: { grid: { color: "rgba(140,180,255,0.08)" }, ticks: { color: "#6b7a94", callback: v => v.toFixed(0) } }
      }
    }
  });
}

function setTimeframe(tf) {
  currentTimeframe = tf;
  document.querySelectorAll(".tf-btn").forEach(b => b.classList.toggle("active", b.textContent === tf));
  buildChart(filterSeries(tf));
}

async function loadBacktest() {
  const data = await fetchJson("/backtest/equity?cost_bps=5");
  renderStats(data.stats);
  allSeries = data.series || [];
  buildChart(filterSeries(currentTimeframe));
}

function buildMonthCurveChart(series) {
  const ctx = document.getElementById("monthCurveChart");
  if (monthCurveChart) monthCurveChart.destroy();

  const data = (series || []).map(p => ({ x: p.date, y: Number(p.equity) }));
  monthCurveChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [{
        data,
        borderColor: "#f59e0b",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.12,
        fill: { target: "origin", above: "rgba(245,158,11,0.10)" }
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => " Equity: " + fmtGBP(c.parsed.y) } }
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "day" },
          grid: { color: "rgba(140,180,255,0.06)" },
          ticks: { color: "#6b7a94", maxTicksLimit: 6 }
        },
        y: {
          grid: { color: "rgba(140,180,255,0.08)" },
          ticks: { color: "#6b7a94", callback: v => "\u00a3" + Number(v).toFixed(0) }
        }
      }
    }
  });
}

function renderContribList(elId, rows) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  if (!rows || !rows.length) {
    el.innerHTML = '<div class="contrib-item"><div class="contrib-left"><b>No data yet</b></div></div>';
    return;
  }
  rows.forEach(r => {
    const pnl = Number(r.pnl_gbp);
    const ret = Number(r.pnl_pct);
    const share = Number(r.contribution_pct);
    const item = document.createElement("div");
    item.className = "contrib-item";
    item.innerHTML = `
      <div class="contrib-left">
        <b>${getTicker(r.ticker)}</b>
        <span class="hc-full">${getName(r.ticker)}</span>
      </div>
      <div class="contrib-right">
        <div style="color:${pnl >= 0 ? '#86efac' : '#fca5a5'}; font-weight:700;">${fmtSignedGBP(pnl)}</div>
        <div>${fmtPct(ret)} return${Number.isFinite(share) ? ` \u00b7 ${(share*100).toFixed(1)}% of PnL` : ""}</div>
      </div>
    `;
    el.appendChild(item);
  });
}

function populateMonthPicker() {
  const sel = document.getElementById("monthPicker");
  const rows = (portfolioHistory || []).filter(r => r.end_date);
  sel.innerHTML = "";
  if (!rows.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No completed months yet";
    sel.appendChild(opt);
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  rows.forEach(r => {
    const opt = document.createElement("option");
    opt.value = r.asof_date;
    opt.textContent = `${r.asof_date} to ${r.end_date}`;
    if ((selectedPerformanceAsOf || (latestPerformance && latestPerformance.asof_date)) === r.asof_date) opt.selected = true;
    sel.appendChild(opt);
  });
}

function renderPortfolioPerformance() {
  const meta = document.getElementById("perfMeta");
  const metrics = document.getElementById("perfMetrics");
  const perf = selectedPerformance || latestPerformance;
  if (!perf || !perf.asof_date) {
    meta.textContent = "No completed month available yet. The first run creates the starting snapshot; performance appears from the next monthly rerun.";
    metrics.innerHTML = "";
    renderContribList("bestContrib", []);
    renderContribList("worstContrib", []);
    buildMonthCurveChart([]);
    return;
  }

  meta.innerHTML = `
    Tracking the real portfolio from <b>${perf.asof_date}</b> to <b>${perf.end_date}</b>.
    Current equity is <b>${fmtGBP(perf.current_equity)}</b> as of <b>${perf.current_snapshot_date}</b>.
  `;

  metrics.innerHTML = `
    <div class="metric-card">
      <div class="metric-label">Current Equity</div>
      <div class="metric-value">${fmtGBP(perf.current_equity)}</div>
      <div class="metric-sub">Started at ${fmtGBP(perf.start_equity)}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Monthly PnL</div>
      <div class="metric-value ${perfClass(perf.pnl_gbp)}">${fmtSignedGBP(perf.pnl_gbp)}</div>
      <div class="metric-sub">${fmtPct(perf.pnl_pct)} vs last month</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">SPY Same Period</div>
      <div class="metric-value ${perfClass(perf.spy_pnl_pct)}">${fmtPct(perf.spy_pnl_pct)}</div>
      <div class="metric-sub">Buy and hold benchmark</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Worst Drawdown</div>
      <div class="metric-value ${perfClass(perf.max_drawdown)}">${fmtPct(perf.max_drawdown)}</div>
      <div class="metric-sub">Fixed-share equity curve</div>
    </div>
  `;

  renderContribList("bestContrib", perf.best_contributors || []);
  renderContribList("worstContrib", perf.worst_contributors || []);
  buildMonthCurveChart(perf.curve || []);
}

// ═══════════════════════════════════════════════════════
// Market conditions box
// ═══════════════════════════════════════════════════════
function renderMarketBox() {
  const el = document.getElementById("marketBox");
  if (!latestRecs || !latestRecs.recs || !latestRecs.recs.length) { el.textContent = "No data yet."; return; }

  const reasons = latestRecs.recs.map(r => r.reasons || "").join(" ");
  let parts = [];

  // Vol scale
  const vsMatch = reasons.match(/Vol scale: ([\d.]+)x/);
  if (vsMatch) {
    const vs = parseFloat(vsMatch[1]);
    if (vs < 0.8) parts.push(`<span class="warn">\u26a0 Market volatility is higher than our 10% target.</span> This means the market is swinging more than usual, so the model has <b>reduced stock exposure</b> to manage risk. Think of it like driving slower in bad weather \u2014 you give up some speed for safety.`);
    else if (vs > 1.1) parts.push(`<span class="ok">\u2713 Markets are calm right now</span>, with volatility below our 10% target. The model has <b>slightly increased stock exposure</b> to take advantage of the stable conditions.`);
    else parts.push(`Markets are behaving normally. Volatility is close to our 10% target, so stock exposure is at its standard level.`);
  }

  // DD tilt
  if (reasons.includes("DD tilt")) {
    const ddMatch = reasons.match(/DD tilt: shifted ([\d.]+)% to bonds/);
    const pct = ddMatch ? ddMatch[1] : "some";
    parts.push(`<br><br><span class="warn">\u26a0 The market has dropped recently from its recent high.</span> As a safety measure, the model has shifted <b>${pct}%</b> of the portfolio from stocks into bonds. This is automatic \u2014 as the market recovers, this will reverse.`);
  }

  // Budget split
  const budgetMatch = reasons.match(/Budgets: eq (\d+)% \/ bd (\d+)% \/ cm (\d+)%/);
  if (budgetMatch) {
    parts.push(`<br><br>Right now the portfolio is split roughly <b>${budgetMatch[1]}% stocks</b>, <b>${budgetMatch[2]}% bonds</b>, and <b>${budgetMatch[3]}% commodities</b>.`);
  }

  el.innerHTML = parts.join("") || "Market conditions are normal. No special adjustments being made.";
}

function allocationRows(source=latestRecs) {
  return [...((source && source.recs) || [])]
    .filter(r => Number(r.target_weight) > 0.000001)
    .sort((a,b) => Number(b.target_weight) - Number(a.target_weight));
}

function actionableRows(source=latestRecs) {
  return allocationRows(source).filter(r => String(r.ticker).toLowerCase() !== "cash" && r.action !== "HOLD_CASH");
}

function allocationTotal(rows) {
  return rows.reduce((s,r) => s + Number(r.target_weight || 0), 0);
}

function displayTotalPct(total) {
  return Math.abs(total - 1) <= 0.0015 ? "100.0%" : `${(total*100).toFixed(1)}%`;
}

function weightMap(source=latestRecs, includeCash=true) {
  const out = {};
  allocationRows(source).forEach(r => {
    const t = String(r.ticker).toLowerCase();
    if (!includeCash && t === "cash") return;
    out[t] = Number(r.target_weight);
  });
  return out;
}

function cashWeight(source=latestRecs) {
  return allocationRows(source).reduce((s,r) => String(r.ticker).toLowerCase() === "cash" ? s + Number(r.target_weight || 0) : s, 0);
}

function investedRows(source=latestRecs) {
  return allocationRows(source).filter(r => String(r.ticker).toLowerCase() !== "cash");
}

function pieWeightForRow(row, source=latestRecs) {
  const invested = investedRows(source).reduce((s,r) => s + Number(r.target_weight || 0), 0);
  return invested > 0 ? Number(row.target_weight || 0) / invested : 0;
}

function changeForTicker(ticker, newWeight) {
  if (!prevRecs || !prevRecs.recs) return "New target";
  const prev = weightMap(prevRecs, true);
  const oldWeight = Number(prev[String(ticker).toLowerCase()] || 0);
  const delta = Number(newWeight) - oldWeight;
  if (oldWeight === 0 && newWeight > 0) return "New holding";
  if (Math.abs(delta) < 0.001) return "Unchanged";
  return `${delta > 0 ? "Up" : "Down"} ${Math.abs(delta*100).toFixed(1)} pts from last run`;
}

// ═══════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════
function renderSummary() {
  const el = document.getElementById("summaryText");
  if (!latestRecs || !latestRecs.recs || !latestRecs.recs.length) { el.textContent = "No recommendations yet. Run the pipeline first."; return; }
  const recs = allocationRows();
  const pv = getPV();
  const pieRows = investedRows();
  const wCash = cashWeight();
  const pieValue = pv * (1 - wCash);
  const cashValue = pv * wCash;
  let nE=0,nB=0,nC=0,wE=0,wB=0,wC=0;
  recs.forEach(r => {
    const w = Number(r.target_weight);
    const b=bucketLabel(r.ticker);
    if(b==="Cash"){}
    else if(b==="Bond ETF"){nB++;wB+=w;}
    else if(b==="Commodity"){nC++;wC+=w;}
    else {nE++;wE+=w;}
  });
  const total = allocationTotal(recs);
  const ok = Math.abs(total - 1) <= 0.005;
  const big = pieRows[0] || recs[0], bigPct = (pieWeightForRow(big)*100).toFixed(1);

  let s = `<b>As of ${latestRecs.asof_date}</b> \u2014 Put <b>${fmtGBP(pieValue)}</b> into the Trading 212 pie and keep <b>${fmtGBP(cashValue)}</b> outside as cash/dry powder. `;
  s += `The pie has <b>${pieRows.length} holdings</b>: ${nE} stock${nE!==1?"s":""}, ${nB} bond${nB!==1?"s":""}, and ${nC} commodit${nC!==1?"ies":"y"}. `;
  s += `The biggest pie position is <b>${getName(big.ticker)}</b> at ${bigPct}% of the pie.`;
  s += `<div class="allocation-check ${ok ? "ok" : "warn"}"><span>Total allocation</span><b>${displayTotalPct(total)}</b></div>`;
  if (!ok) s += `<div class="small" style="color:#fbbf24;margin-top:8px;">Raw model total is ${(total*100).toFixed(3)}%, so check the latest pipeline run before trading.</div>`;
  s += `<div class="allocation-breakdown">
    <div class="allocation-chip">Pie amount <b>${fmtGBP(pieValue)}</b></div>
    <div class="allocation-chip">Outside cash <b>${fmtGBP(cashValue)}</b></div>
    <div class="allocation-chip">Invested account share <b>${((1-wCash)*100).toFixed(1)}%</b></div>
    <div class="allocation-chip">Cash account share <b>${(wCash*100).toFixed(1)}%</b></div>
  </div>`;
  el.innerHTML = s;
}

// ═══════════════════════════════════════════════════════
// Holdings cards
// ═══════════════════════════════════════════════════════
function renderHoldings() {
  const c = document.getElementById("holdingsCards"); c.innerHTML = "";
  if (!latestRecs || !latestRecs.recs) return;
  const recs = investedRows();
  const pv = getPV(), pm = priceMap();
  const wCash = cashWeight();
  const pieValue = pv * (1 - wCash);
  const cashValue = pv * wCash;
  document.getElementById("holdingsExplainer").textContent = `Put ${fmtGBP(pieValue)} into this pie. Keep ${fmtGBP(cashValue)} outside as cash. Pie weights below add to 100.0%.`;
  const maxW = Math.max(...recs.map(r => pieWeightForRow(r)), 0.01);

  recs.forEach(r => {
    const ticker = String(r.ticker).toLowerCase();
    const w=pieWeightForRow(r), accountW=Number(r.target_weight), px=Number(r.price || pm[ticker] || 0), val=accountW*pv, sh=Number(r.shares || (px?val/px:0));
    const reason = humaniseReasons(r.ticker, r.reasons, w);
    const card = document.createElement("div"); card.className = "holding-card";
    const shareText = px ? `~${fmtNum(sh)} shares at ${fmtGBP(px)}` : "No latest price available";
    card.innerHTML = `
      <div class="hc-top">
        <div><span class="hc-name">${getTicker(r.ticker)}</span> <span class="hc-full">${getName(r.ticker)} \u00b7 ${bucketLabel(r.ticker)}</span></div>
        <span class="hc-weight">${(w*100).toFixed(1)}%</span>
      </div>
      <div class="hc-bar"><div class="hc-bar-fill" style="width:${Math.min(w/maxW*100,100).toFixed(0)}%"></div></div>
      <div class="hc-details">${fmtGBP(val)} \u2014 ${shareText}</div>
      <div class="hc-meta">
        <span class="mini-pill">Pie weight ${(w*100).toFixed(1)}%</span>
        <span class="mini-pill">Account weight ${(accountW*100).toFixed(1)}%</span>
        <span class="mini-pill">${changeForTicker(r.ticker, accountW)}</span>
      </div>
      ${reason?`<div class="hc-reason">${reason}</div>`:""}
    `;
    c.appendChild(card);
  });
}

// ═══════════════════════════════════════════════════════
// Changes
// ═══════════════════════════════════════════════════════
function renderChanges() {
  const c = document.getElementById("changesList"); c.innerHTML = "";
  if (!latestRecs || !latestRecs.recs) return;
  const latest=weightMap(latestRecs, false), prev=weightMap(prevRecs, false);
  const changes=[];
  new Set([...Object.keys(latest),...Object.keys(prev)]).forEach(t=>{
    const wN=latest[t]||0, wO=prev[t]||0;
    if(wN>0&&wO===0) changes.push({ticker:t,type:"new",label:"New",desc:`Added at ${(wN*100).toFixed(0)}%`});
    else if(wN===0&&wO>0) changes.push({ticker:t,type:"exit",label:"Removed",desc:`Was ${(wO*100).toFixed(0)}%`});
    else if(Math.abs(wN-wO)>=0.01){const dir=wN>wO?"buy":"sell",verb=wN>wO?"Increased":"Reduced";changes.push({ticker:t,type:dir,label:verb,desc:`${(wO*100).toFixed(1)}% \u2192 ${(wN*100).toFixed(1)}%`});}
  });
  const ex=document.getElementById("changesExplainer");
  if(!prevRecs){ex.textContent="First month \u2014 everything is new.";return;}
  if(!changes.length){ex.textContent=`No significant changes since ${prevRecs.asof_date}.`;return;}
  const nN=changes.filter(x=>x.type==="new").length,nE=changes.filter(x=>x.type==="exit").length,nA=changes.filter(x=>x.type==="buy"||x.type==="sell").length;
  const p=[];if(nN)p.push(nN+" new");if(nE)p.push(nE+" removed");if(nA)p.push(nA+" adjusted");
  ex.textContent=`Since ${prevRecs.asof_date}: ${p.join(", ")}.`;
  const order={new:0,exit:1,buy:2,sell:3};
  changes.sort((a,b)=>(order[a.type]||9)-(order[b.type]||9));
  changes.forEach(ch=>{
    const row=document.createElement("div");row.className=`change-row ${ch.type}`;
    row.innerHTML=`<span class="badge ${ch.type}">${ch.label}</span> <b>${getTicker(ch.ticker)}</b> <span class="hc-full">${getName(ch.ticker)}</span> <span class="muted small">${ch.desc}</span>`;
    c.appendChild(row);
  });
}

// ═══════════════════════════════════════════════════════
// Actions
// ═══════════════════════════════════════════════════════
function renderActions() {
  const c=document.getElementById("actionsList");c.innerHTML="";
  if(!latestRecs||!latestRecs.recs)return;
  if(latestTrades && latestTrades.trades && latestTrades.trades.length){
    latestTrades.trades.forEach(t=>{
      const action=String(t.trade_action || "").toUpperCase();
      const shares=Math.abs(Number(t.shares_delta || 0));
      const notional=Math.abs(Number(t.est_notional || 0));
      const row=document.createElement("div");row.className="action-row";
      const col=action==="BUY"?"#86efac":"#fca5a5";
      row.innerHTML=`<div class="ar-left"><span class="badge ${action.toLowerCase()}">${action}</span> <b>${getTicker(t.ticker)}</b> <span class="hc-full">${getName(t.ticker)}</span></div><div class="ar-right">~${fmtNum(shares)} shares \u00b7 <b style="color:${col}">${fmtGBP(notional)}</b></div>`;
      c.appendChild(row);
    });
    return;
  }
  const pm=priceMap(),pv=getPV(),latest={},prev={};
  actionableRows(latestRecs).forEach(r=>latest[String(r.ticker).toLowerCase()]=Number(r.target_weight));
  actionableRows(prevRecs).forEach(r=>prev[String(r.ticker).toLowerCase()]=Number(r.target_weight));
  const trades=[];
  new Set([...Object.keys(latest),...Object.keys(prev)]).forEach(t=>{
    const wN=latest[t]||0,wO=prev[t]||0,wD=wN-wO;if(Math.abs(wD)<0.01)return;
    const px=pm[t];if(!px)return;const notional=Math.abs(wD)*pv;if(notional<5)return;
    trades.push({ticker:t,action:wD>0?"BUY":"SELL",shares:notional/px,notional});
  });
  trades.sort((a,b)=>b.notional-a.notional);
  if(!trades.length){c.innerHTML='<div class="action-row"><span class="muted">No trades needed this month.</span></div>';return;}
  trades.forEach(t=>{
    const row=document.createElement("div");row.className="action-row";
    const col=t.action==="BUY"?"#86efac":"#fca5a5";
    row.innerHTML=`<div class="ar-left"><span class="badge ${t.action.toLowerCase()}">${t.action}</span> <b>${getTicker(t.ticker)}</b> <span class="hc-full">${getName(t.ticker)}</span></div><div class="ar-right">~${fmtNum(t.shares)} shares \u00b7 <b style="color:${col}">${fmtGBP(t.notional)}</b></div>`;
    c.appendChild(row);
  });
}

// ═══════════════════════════════════════════════════════
// Commodity Scout tab
// ═══════════════════════════════════════════════════════
function commodityRows(includeCash=true) {
  const rows = [...((commodityLatest && commodityLatest.holdings) || [])]
    .filter(r => includeCash || String(r.ticker).toLowerCase() !== "cash")
    .sort((a,b) => Number(b.target_weight || 0) - Number(a.target_weight || 0));
  return rows;
}

function renderCommoditySummary() {
  const el = document.getElementById("commoditySummary");
  if (!commodityLatest || !commodityLatest.asof_date) {
    el.innerHTML = `No Commodity Scout run yet. Run <code>python run_commodity_pipeline.py</code> to build this separate pie.`;
    return;
  }
  const summary = commodityLatest.summary || {};
  const total = Number(summary.total_weight || 0);
  const cashW = Number(summary.cash_weight || 0);
  const cashVal = cashW * getPV();
  const ok = Math.abs(total - 1) <= 0.005;
  const noncash = commodityRows(false);
  const top = noncash[0];
  let s = `<b>As of ${commodityLatest.asof_date}</b> — Commodity Scout picked <b>${summary.non_cash_positions || noncash.length}</b> invested positions. `;
  if (top) s += `The largest commodity is <b>${getName(top.ticker)}</b> at ${(Number(top.target_weight)*100).toFixed(1)}% of the total commodity account. `;
  if (cashW > 0) s += `<b>Keep ${fmtGBP(cashVal)} outside the pie</b> as cash/dry powder. `;
  s += `<div class="allocation-check ${ok ? "ok" : "warn"}"><span>Total allocation including outside cash</span><b>${displayTotalPct(total)}</b></div>`;
  s += `<div class="pie-chip-grid">
    <div class="pie-chip">Invested inside pie <b>${((1-cashW)*100).toFixed(1)}%</b></div>
    <div class="pie-chip">Outside cash <b>${(cashW*100).toFixed(1)}%</b></div>
    <div class="pie-chip">Smallest holding <b>${((summary.min_non_cash_weight||0)*100).toFixed(1)}%</b></div>
    <div class="pie-chip">Largest holding <b>${((summary.max_non_cash_weight||0)*100).toFixed(1)}%</b></div>
  </div>`;
  el.innerHTML = s;
}

function renderCommodityHoldings() {
  const wrap = document.getElementById("commodityHoldings");
  wrap.innerHTML = "";
  const rows = commodityRows(true);
  if (!rows.length) {
    document.getElementById("commodityHoldingsExplainer").textContent = "No commodity holdings yet.";
    return;
  }
  document.getElementById("commodityHoldingsExplainer").textContent = `${rows.length} allocation lines. Non-cash commodities must be at least 10% and no more than 35%.`;
  const maxW = Math.max(...rows.map(r => Number(r.target_weight || 0)), 0.01);
  rows.forEach(r => {
    const ticker = String(r.ticker).toLowerCase();
    const isCash = ticker === "cash";
    const w = Number(r.target_weight || 0);
    const invW = Number(r.invested_weight || 0);
    const px = Number(r.price || 0);
    const val = w * getPV();
    const shares = isCash ? 0 : (px ? val / px : Number(r.shares || 0));
    const card = document.createElement("div");
    card.className = `commodity-card${isCash ? " cash" : ""}`;
    card.innerHTML = `
      <div class="commodity-top">
        <div>
          <b>${getTicker(r.ticker)}</b>
          <span class="hc-full">${isCash ? "Cash outside pie" : `${r.trading212_name || getName(r.ticker)} · ${r.commodity || "commodity"}`}</span>
        </div>
        <span class="commodity-weight">${(w*100).toFixed(1)}%</span>
      </div>
      <div class="hc-bar"><div class="hc-bar-fill" style="width:${Math.min(w/maxW*100,100).toFixed(0)}%; background:linear-gradient(90deg,#f59e0b,#fbbf24);"></div></div>
      <div class="hc-details">${fmtGBP(val)} — ${isCash ? "keep outside the Trading 212 pie" : `~${fmtNum(shares)} shares at ${fmtGBP(px)} · pie weight ${(invW*100).toFixed(1)}%`}</div>
      <div class="hc-meta">
        <span class="mini-pill">Confidence ${((Number(r.confidence||0))*100).toFixed(0)}%</span>
        ${!isCash ? `<span class="mini-pill">Trading 212: ${r.trading212_ticker || getTicker(r.ticker)}</span>` : ""}
      </div>
      <div class="hc-reason">${r.reasons || "No explanation saved."}</div>
    `;
    wrap.appendChild(card);
  });
}

function renderCommodityPieWeights() {
  const el = document.getElementById("commodityPieWeights");
  const rows = commodityRows(false);
  if (!rows.length) {
    el.innerHTML = '<div class="action-row"><span class="muted">No invested commodity pie rows yet.</span></div>';
    return;
  }
  const total = rows.reduce((s,r) => s + Number(r.invested_weight || 0), 0);
  let html = `<div class="allocation-check ${Math.abs(total-1)<=0.005 ? "ok" : "warn"}"><span>Trading 212 pie total</span><b>${displayTotalPct(total)}</b></div>`;
  rows.forEach(r => {
    html += `<div class="action-row"><div class="ar-left"><b>${r.trading212_ticker || getTicker(r.ticker)}</b><span class="hc-full">${r.trading212_name || getName(r.ticker)}</span></div><div class="ar-right"><b style="color:#fbbf24;">${(Number(r.invested_weight||0)*100).toFixed(1)}%</b></div></div>`;
  });
  el.innerHTML = html;
}

function renderCommodityTrades() {
  const el = document.getElementById("commodityTrades");
  const trades = (commodityTrades && commodityTrades.trades) || [];
  if (!trades.length) {
    el.innerHTML = '<div class="action-row"><span class="muted">No commodity trades needed yet, or no previous commodity snapshot exists.</span></div>';
    return;
  }
  el.innerHTML = "";
  trades.forEach(t => {
    const action = String(t.trade_action || "").toUpperCase();
    const col = action === "BUY" ? "#86efac" : "#fca5a5";
    const row = document.createElement("div");
    row.className = "action-row";
    row.innerHTML = `<div class="ar-left"><span class="badge ${action.toLowerCase()}">${action}</span><b>${getTicker(t.ticker)}</b><span class="hc-full">${getName(t.ticker)}</span></div><div class="ar-right">~${fmtNum(Math.abs(Number(t.shares_delta || 0)))} shares · <b style="color:${col}">${fmtGBP(Math.abs(Number(t.est_notional || 0)))}</b></div>`;
    el.appendChild(row);
  });
}

function renderCommodityNews() {
  const el = document.getElementById("commodityNews");
  const features = (commodityNews && commodityNews.features) || [];
  const articles = (commodityNews && commodityNews.articles) || [];
  if (!features.length && !articles.length) {
    el.innerHTML = '<div class="news-row">No commodity news stored yet. The model can still run from price and macro data, then news will appear after RSS fetch succeeds.</div>';
    return;
  }
  let html = "";
  features.slice(0, 6).forEach(f => {
    html += `<div class="news-row"><b>${getTicker(f.ticker)}</b> ${f.commodity || ""}: ${Number(f.news_count_7d||0).toFixed(0)} stories in 7d · sentiment ${Number(f.sent_mean_7d||0).toFixed(2)} · shock ${Number(f.sent_shock||0).toFixed(2)}</div>`;
  });
  articles.slice(0, 6).forEach(a => {
    html += `<div class="news-row"><b>${a.commodity || getTicker(a.ticker)}</b> ${a.headline || ""}<div class="muted">${a.source || ""} · ${a.published_at || ""} · sentiment ${Number(a.sentiment || 0).toFixed(2)}</div></div>`;
  });
  el.innerHTML = html;
}

function renderCommodityBacktest() {
  const statsEl = document.getElementById("commodityStats");
  const data = commodityBacktest || {stats:{}, series:[], benchmark:[]};
  renderStatsLike(statsEl, data.stats || {});
  const ctx = document.getElementById("commodityChart");
  if (!ctx) return;
  if (commodityChart) commodityChart.destroy();
  const series = data.series || [];
  const benchmark = data.benchmark || [];
  commodityChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        { label: "Commodity Scout", data: series.map(p => ({x:p.date, y:Number(p.equity)*100})), borderColor: "#f59e0b", borderWidth: 2, pointRadius: 0, tension: 0.1 },
        { label: data.benchmark_name || "DBC", data: benchmark.map(p => ({x:p.date, y:Number(p.equity)*100})), borderColor: "#64748b", borderWidth: 1.5, pointRadius: 0, tension: 0.1 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#8b98b0" } } },
      scales: {
        x: { type: "time", grid: { color: "rgba(140,180,255,0.06)" }, ticks: { color: "#6b7a94", maxTicksLimit: 6 } },
        y: { grid: { color: "rgba(140,180,255,0.08)" }, ticks: { color: "#6b7a94", callback: v => Number(v).toFixed(0) } }
      }
    }
  });
}

function renderStatsLike(el, stats) {
  el.innerHTML = `
    <span class="pill">Period <b>${stats.period_label || "Recent"}</b></span>
    <span class="pill">Annual return <b>${fmtPct(stats.cagr)}</b></span>
    <span class="pill">Risk <b>${fmtPct(stats.vol)}</b></span>
    <span class="pill">Sharpe <b>${stats.sharpe!=null?Number(stats.sharpe).toFixed(2):"n/a"}</b></span>
    <span class="pill">Worst drop <b>${fmtPct(stats.max_drawdown)}</b></span>
    ${stats.note ? `<div class="muted small" style="width:100%; margin-top:2px;">${stats.note}</div>` : ""}
  `;
}

function renderCommodityAll() {
  renderCommoditySummary();
  renderCommodityHoldings();
  renderCommodityPieWeights();
  renderCommodityTrades();
  renderCommodityNews();
  renderCommodityBacktest();
}

// ═══════════════════════════════════════════════════════
// Downloads
// ═══════════════════════════════════════════════════════
function downloadPieCSV() {
  if(!latestRecs||!latestRecs.recs)return;
  const rows=investedRows();
  const pv=getPV();
  const wCash = cashWeight();
  const pieValue = pv * (1 - wCash);
  let csv="Ticker,Name,Type,PieWeight,PiePercent,AccountWeight,EstimatedValue\n";
  rows.forEach(r=>{const accountW=Number(r.target_weight);const w=pieWeightForRow(r);csv+=`${csvCell(r.ticker)},${csvCell(getName(r.ticker))},${csvCell(bucketLabel(r.ticker))},${w.toFixed(6)},${(w*100).toFixed(2)},${accountW.toFixed(6)},${(w*pieValue).toFixed(2)}\n`;});
  csv+=`${csvCell("OUTSIDE_CASH")},${csvCell("Keep outside Trading 212 pie")},${csvCell("Cash")},0.000000,0.00,${wCash.toFixed(6)},${(pv*wCash).toFixed(2)}\n`;
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download=`trading212_pie_${latestRecs.asof_date||"latest"}.csv`;document.body.appendChild(a);a.click();document.body.removeChild(a);
}

function downloadCommodityCSV() {
  window.location="/commodities/export/trading212.csv";
}

// ═══════════════════════════════════════════════════════
// Data loading
// ═══════════════════════════════════════════════════════
async function loadPrices(){latestPrices=await fetchJson("/prices/latest");}
async function loadRecs(){latestRecs=await fetchJson("/model/holdings/latest");}
async function loadRecHistory(){
  const d=await fetchJson("/model/holdings/history?limit=2");const s=d.snapshots||[];
  if(s[0])latestRecs={asof_date:s[0].asof_date,recs:s[0].recs};
  prevRecs=s[1]?{asof_date:s[1].asof_date,recs:s[1].recs}:null;
}
async function loadTrades(){latestTrades=await fetchJson("/model/trades/latest");}
async function loadCommodityLatest(){commodityLatest=await fetchJson("/commodities/latest");}
async function loadCommodityNews(){commodityNews=await fetchJson("/commodities/news?limit=40");}
async function loadCommodityTrades(){commodityTrades=await fetchJson("/commodities/trades/latest");}
async function loadCommodityBacktest(){commodityBacktest=await fetchJson("/commodities/backtest?cost_bps=5&start_date=2015-01-01");}
async function loadPortfolioHistory(){
  const d = await fetchJson("/portfolio/history?limit=60");
  portfolioHistory = d.history || [];
}
async function loadPortfolioPerformance(asof=null){
  const url = asof ? `/portfolio/performance?asof_date=${encodeURIComponent(asof)}` : "/portfolio/performance/latest";
  const data = await fetchJson(url);
  if (!asof) latestPerformance = data;
  selectedPerformance = data;
  selectedPerformanceAsOf = data && data.asof_date ? data.asof_date : null;
}
function renderAll(){
  document.getElementById("pvLabel").textContent=String(getPV());
  renderSummary();renderHoldings();renderChanges();renderActions();renderMarketBox();populateMonthPicker();renderPortfolioPerformance();
  if (document.getElementById("commoditiesTab").classList.contains("active")) renderCommodityAll();
}
async function reloadAll(){
  await Promise.all([loadPrices(),loadRecs(),loadRecHistory(),loadTrades(),loadPortfolioHistory(),loadCommodityLatest(),loadCommodityNews(),loadCommodityTrades(),loadCommodityBacktest()]);
  await loadPortfolioPerformance(selectedPerformanceAsOf);
  if (!selectedPerformanceAsOf && latestPerformance && latestPerformance.asof_date) {
    selectedPerformanceAsOf = latestPerformance.asof_date;
  }
  renderAll();
  await loadBacktest();
}
document.getElementById("portfolioValue").addEventListener("input",()=>renderAll());
document.getElementById("monthPicker").addEventListener("change", async (e) => {
  const value = e.target.value || null;
  await loadPortfolioPerformance(value);
  renderAll();
});
reloadAll();
</script>
</body>
</html>
    """
    default_pv = f"{_current_portfolio_value():.2f}".rstrip("0").rstrip(".")
    html = html.replace("__DEFAULT_PORTFOLIO_VALUE__", default_pv)
    return HTMLResponse(content=html)


# ── Keep these for programmatic access ──

@app.get("/portfolio/holdings/latest")
def latest_holdings():
    with get_conn() as conn:
        asof = _latest_asof_on_or_before_prices(conn, "model_holdings")
        if asof is None:
            return {"asof_date": None, "holdings": []}
        df = pd.read_sql_query(
            "SELECT * FROM model_holdings WHERE asof_date = ? ORDER BY value DESC",
            conn, params=(asof,),
        )
    return {"asof_date": asof, "holdings": df.to_dict(orient="records")}


@app.get("/portfolio/trades/latest")
def latest_trades():
    with get_conn() as conn:
        asof = _latest_asof_on_or_before_prices(conn, "model_trades")
        if asof is None:
            return {"asof_date": None, "trades": []}
        df = pd.read_sql_query(
            "SELECT * FROM model_trades WHERE asof_date = ? ORDER BY est_notional DESC",
            conn, params=(asof,),
        )
    return {"asof_date": asof, "trades": df.to_dict(orient="records")}
