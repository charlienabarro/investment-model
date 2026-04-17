from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .db import get_conn

DEFAULT_STARTING_EQUITY = 10000.0
LEARNING_DECAY = 0.85
LEARNING_IDLE_DECAY = 0.97
# Keep feedback small: ML momentum features already see recent winners, and a
# larger overlay can double-count that momentum or chase reversals.
LEARNING_RATE = 0.05
LEARNING_CAP = 0.35


def _iso_date(value) -> str:
    return pd.to_datetime(value).date().isoformat()


def _iso_datetime(value=None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    return pd.to_datetime(value, utc=True).isoformat()


def _json_safe_value(value):
    if isinstance(value, dict):
        return {k: _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(v) for v in value]
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _records_json_safe(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []
    return [_json_safe_value(row) for row in df.to_dict(orient="records")]


def _price_map_on_or_before(conn, asof_date: str, tickers: List[str]) -> Dict[str, float]:
    clean = sorted({str(t).lower() for t in tickers if str(t).lower() != "cash"})
    if not clean:
        return {"cash": 1.0}

    placeholders = ",".join(["?"] * len(clean))
    params = [*clean, asof_date]
    df = pd.read_sql_query(
        f"""
        SELECT p1.ticker, p1.close
        FROM prices_daily p1
        JOIN (
            SELECT ticker, MAX(date) AS max_date
            FROM prices_daily
            WHERE ticker IN ({placeholders}) AND date <= ?
            GROUP BY ticker
        ) p2
        ON p1.ticker = p2.ticker AND p1.date = p2.max_date
        """,
        conn,
        params=params,
    )
    out = dict(zip(df["ticker"].astype(str).str.lower(), df["close"].astype(float)))
    out["cash"] = 1.0
    return out


def _single_close_on_or_before(conn, ticker: str, asof_date: str) -> Optional[float]:
    if str(ticker).lower() == "cash":
        return 1.0
    row = conn.execute(
        """
        SELECT close
        FROM prices_daily
        WHERE ticker = ? AND date <= ?
        ORDER BY date DESC
        LIMIT 1
        """,
        (str(ticker).lower(), asof_date),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _latest_snapshot_before(conn, asof_date: str) -> Optional[str]:
    row = conn.execute(
        """
        SELECT MAX(asof_date)
        FROM portfolio_snapshots
        WHERE asof_date < ?
        """,
        (asof_date,),
    ).fetchone()
    return row[0] if row and row[0] else None


def _latest_completed_equity(conn, asof_date: str) -> Optional[float]:
    row = conn.execute(
        """
        SELECT end_equity
        FROM portfolio_performance
        WHERE end_date <= ?
        ORDER BY end_date DESC, asof_date DESC
        LIMIT 1
        """,
        (asof_date,),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _latest_snapshot_asof(conn, max_date: Optional[str] = None) -> Optional[str]:
    if max_date is None:
        row = conn.execute("SELECT MAX(asof_date) FROM portfolio_snapshots").fetchone()
    else:
        row = conn.execute(
            """
            SELECT MAX(asof_date)
            FROM portfolio_snapshots
            WHERE asof_date <= ?
            """,
            (_iso_date(max_date),),
        ).fetchone()
    return row[0] if row and row[0] else None


def _latest_price_date(conn) -> Optional[str]:
    row = conn.execute("SELECT MAX(date) FROM prices_daily").fetchone()
    return row[0] if row and row[0] else None


def _load_snapshot_holdings(conn, asof_date: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT asof_date, ticker, shares, price, weight, value
        FROM portfolio_snapshot_holdings
        WHERE asof_date = ?
        ORDER BY value DESC, ticker ASC
        """,
        conn,
        params=(asof_date,),
    )
    if not df.empty:
        df["ticker"] = df["ticker"].astype(str).str.lower()
        for col in ["shares", "price", "weight", "value"]:
            df[col] = df[col].astype(float)
    return df


def _build_daily_curve(conn, prev_asof_date: str, next_asof_date: str) -> pd.DataFrame:
    prev_asof_date = _iso_date(prev_asof_date)
    next_asof_date = _iso_date(next_asof_date)
    if prev_asof_date > next_asof_date:
        return pd.DataFrame(columns=["date", "equity"])

    holdings = _load_snapshot_holdings(conn, prev_asof_date)
    if holdings.empty:
        return pd.DataFrame(columns=["date", "equity"])

    tickers = holdings["ticker"].astype(str).tolist()
    tickers_no_cash = [t for t in tickers if t != "cash"]

    dates = pd.read_sql_query(
        """
        SELECT DISTINCT date
        FROM prices_daily
        WHERE date >= ? AND date <= ?
        ORDER BY date ASC
        """,
        conn,
        params=(prev_asof_date, next_asof_date),
    )
    if dates.empty:
        dates = pd.DataFrame({"date": [prev_asof_date, next_asof_date]})
    cal = pd.to_datetime(dates["date"]).drop_duplicates().sort_values()
    cal = pd.Index(sorted(set(cal.tolist() + [pd.to_datetime(prev_asof_date), pd.to_datetime(next_asof_date)])))

    if tickers_no_cash:
        placeholders = ",".join(["?"] * len(tickers_no_cash))
        px_raw = pd.read_sql_query(
            f"""
            SELECT date, ticker, close
            FROM prices_daily
            WHERE ticker IN ({placeholders}) AND date >= ? AND date <= ?
            ORDER BY date ASC, ticker ASC
            """,
            conn,
            params=[*tickers_no_cash, prev_asof_date, next_asof_date],
        )
        px = px_raw.pivot(index="date", columns="ticker", values="close") if not px_raw.empty else pd.DataFrame()
        if not px.empty:
            px.index = pd.to_datetime(px.index)
            px = px.sort_index().reindex(cal).ffill()
        else:
            px = pd.DataFrame(index=cal, columns=tickers_no_cash, dtype=float)
    else:
        px = pd.DataFrame(index=cal)

    for _, row in holdings.iterrows():
        tkr = row["ticker"]
        if tkr == "cash":
            px["cash"] = 1.0
            continue
        if tkr not in px.columns:
            px[tkr] = np.nan
        px.loc[pd.to_datetime(prev_asof_date), tkr] = float(row["price"])

    px = px.sort_index().ffill().fillna(0.0)
    shares = holdings.set_index("ticker")["shares"].astype(float)
    curve = px.mul(shares.reindex(px.columns).fillna(0.0), axis=1).sum(axis=1)
    return pd.DataFrame({"date": px.index, "equity": curve.values})


def save_portfolio_snapshot(
    asof_date: str,
    default_starting_equity: float = DEFAULT_STARTING_EQUITY,
    starting_equity_override: Optional[float] = None,
) -> Optional[dict]:
    asof_date = _iso_date(asof_date)
    created_at = _iso_datetime()

    with get_conn() as conn:
        if starting_equity_override is not None:
            start_equity = float(starting_equity_override)
        else:
            start_equity = _latest_completed_equity(conn, asof_date)
            if start_equity is None:
                start_equity = float(default_starting_equity)

        model_holdings = pd.read_sql_query(
            """
            SELECT ticker, target_weight, price, shares, value
            FROM model_holdings
            WHERE asof_date = ?
            ORDER BY value DESC, ticker ASC
            """,
            conn,
            params=(asof_date,),
        )
        if model_holdings.empty:
            return None

        model_holdings["ticker"] = model_holdings["ticker"].astype(str).str.lower()
        weights = pd.to_numeric(model_holdings["target_weight"], errors="coerce").fillna(0.0)
        w_sum = float(weights.sum())
        if w_sum <= 0:
            return None
        model_holdings["weight"] = weights / w_sum

        price_map = _price_map_on_or_before(conn, asof_date, model_holdings["ticker"].tolist())
        rows = []
        for _, row in model_holdings.iterrows():
            ticker = row["ticker"]
            price = float(price_map.get(ticker, row["price"] if pd.notna(row["price"]) else 0.0))
            if ticker != "cash" and price <= 0:
                continue
            weight = float(row["weight"])
            value = float(start_equity) * weight
            shares = value / price if price > 0 else 0.0
            rows.append((asof_date, ticker, shares, price, weight, value))

        conn.execute(
            """
            INSERT INTO portfolio_snapshots (asof_date, starting_equity, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(asof_date) DO UPDATE SET
                starting_equity = excluded.starting_equity,
                created_at = excluded.created_at
            """,
            (asof_date, float(start_equity), created_at),
        )

        conn.execute("DELETE FROM portfolio_snapshot_holdings WHERE asof_date = ?", (asof_date,))
        conn.executemany(
            """
            INSERT INTO portfolio_snapshot_holdings (asof_date, ticker, shares, price, weight, value)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(asof_date, ticker) DO UPDATE SET
                shares = excluded.shares,
                price = excluded.price,
                weight = excluded.weight,
                value = excluded.value
            """,
            rows,
        )

        return {
            "asof_date": asof_date,
            "starting_equity": float(start_equity),
            "holdings": len(rows),
        }


def compute_month_performance(prev_asof_date: str, next_asof_date: str) -> Optional[dict]:
    prev_asof_date = _iso_date(prev_asof_date)
    next_asof_date = _iso_date(next_asof_date)
    created_at = _iso_datetime()

    with get_conn() as conn:
        holdings = _load_snapshot_holdings(conn, prev_asof_date)
        if holdings.empty:
            return None

        end_price_map = _price_map_on_or_before(conn, next_asof_date, holdings["ticker"].tolist())
        holdings["end_price"] = holdings["ticker"].map(end_price_map).fillna(0.0).astype(float)
        holdings.loc[holdings["ticker"] == "cash", "end_price"] = 1.0

        holdings["start_value"] = holdings["shares"] * holdings["price"]
        holdings["end_value"] = holdings["shares"] * holdings["end_price"]
        holdings["pnl_gbp"] = holdings["end_value"] - holdings["start_value"]
        holdings["pnl_pct"] = np.where(
            holdings["start_value"].abs() > 1e-12,
            holdings["end_value"] / holdings["start_value"] - 1.0,
            0.0,
        )

        start_equity = float(holdings["start_value"].sum())
        end_equity = float(holdings["end_value"].sum())
        pnl_gbp = end_equity - start_equity
        pnl_pct = (end_equity / start_equity - 1.0) if abs(start_equity) > 1e-12 else 0.0

        if abs(pnl_gbp) > 1e-12:
            holdings["contribution_pct"] = holdings["pnl_gbp"] / pnl_gbp
        else:
            holdings["contribution_pct"] = 0.0

        curve = _build_daily_curve(conn, prev_asof_date, next_asof_date)
        if curve.empty:
            max_drawdown = 0.0
        else:
            curve["running_max"] = curve["equity"].cummax()
            curve["drawdown"] = curve["equity"] / curve["running_max"] - 1.0
            max_drawdown = float(curve["drawdown"].min())

        spy_start = _single_close_on_or_before(conn, "spy.us", prev_asof_date)
        spy_end = _single_close_on_or_before(conn, "spy.us", next_asof_date)
        spy_pnl_pct = None
        if spy_start and spy_end and spy_start > 0:
            spy_pnl_pct = float(spy_end / spy_start - 1.0)

        conn.execute(
            """
            INSERT INTO portfolio_performance (
                asof_date, end_date, start_equity, end_equity, pnl_gbp, pnl_pct,
                max_drawdown, spy_start, spy_end, spy_pnl_pct, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asof_date) DO UPDATE SET
                end_date = excluded.end_date,
                start_equity = excluded.start_equity,
                end_equity = excluded.end_equity,
                pnl_gbp = excluded.pnl_gbp,
                pnl_pct = excluded.pnl_pct,
                max_drawdown = excluded.max_drawdown,
                spy_start = excluded.spy_start,
                spy_end = excluded.spy_end,
                spy_pnl_pct = excluded.spy_pnl_pct,
                created_at = excluded.created_at
            """,
            (
                prev_asof_date,
                next_asof_date,
                float(start_equity),
                float(end_equity),
                float(pnl_gbp),
                float(pnl_pct),
                float(max_drawdown),
                float(spy_start) if spy_start is not None else None,
                float(spy_end) if spy_end is not None else None,
                float(spy_pnl_pct) if spy_pnl_pct is not None else None,
                created_at,
            ),
        )

        contrib_rows = [
            (
                prev_asof_date,
                str(row["ticker"]).lower(),
                float(row["pnl_gbp"]),
                float(row["pnl_pct"]),
                float(row["contribution_pct"]),
            )
            for _, row in holdings.iterrows()
        ]
        conn.executemany(
            """
            INSERT INTO portfolio_contrib (asof_date, ticker, pnl_gbp, pnl_pct, contribution_pct)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(asof_date, ticker) DO UPDATE SET
                pnl_gbp = excluded.pnl_gbp,
                pnl_pct = excluded.pnl_pct,
                contribution_pct = excluded.contribution_pct
            """,
            contrib_rows,
        )

        return {
            "asof_date": prev_asof_date,
            "end_date": next_asof_date,
            "start_equity": float(start_equity),
            "end_equity": float(end_equity),
            "pnl_gbp": float(pnl_gbp),
            "pnl_pct": float(pnl_pct),
            "max_drawdown": float(max_drawdown),
            "spy_start": float(spy_start) if spy_start is not None else None,
            "spy_end": float(spy_end) if spy_end is not None else None,
            "spy_pnl_pct": float(spy_pnl_pct) if spy_pnl_pct is not None else None,
        }


def update_learning_from_performance(prev_asof_date: str, next_asof_date: str) -> dict:
    prev_asof_date = _iso_date(prev_asof_date)
    next_asof_date = _iso_date(next_asof_date)
    updated_at = _iso_datetime()

    with get_conn() as conn:
        perf = pd.read_sql_query(
            """
            SELECT asof_date, end_date, start_equity
            FROM portfolio_performance
            WHERE asof_date = ? AND end_date = ?
            """,
            conn,
            params=(prev_asof_date, next_asof_date),
        )
        if perf.empty:
            return {"updated": 0, "effective_asof": next_asof_date}

        start_equity = float(perf["start_equity"].iloc[0])
        contrib = pd.read_sql_query(
            """
            SELECT ticker, pnl_gbp, pnl_pct, contribution_pct
            FROM portfolio_contrib
            WHERE asof_date = ?
            """,
            conn,
            params=(prev_asof_date,),
        )
        contrib["ticker"] = contrib["ticker"].astype(str).str.lower()

        existing = pd.read_sql_query(
            """
            SELECT ticker, score_adjustment, last_signal
            FROM learned_params
            """,
            conn,
        )
        current_adj = {
            str(row["ticker"]).lower(): float(row["score_adjustment"])
            for _, row in existing.iterrows()
        }
        current_signal = {
            str(row["ticker"]).lower(): float(row["last_signal"])
            for _, row in existing.iterrows()
        }

        contrib_map = {str(row["ticker"]).lower(): row for _, row in contrib.iterrows()}
        all_tickers = sorted(set(current_adj.keys()) | set(contrib_map.keys()))
        rows = []

        for ticker in all_tickers:
            old_adj = float(current_adj.get(ticker, 0.0))
            if ticker in contrib_map:
                row = contrib_map[ticker]
                pnl_pct = float(row["pnl_pct"])
                impact_pct = float(row["pnl_gbp"]) / start_equity if abs(start_equity) > 1e-12 else 0.0
                signal = 0.7 * float(np.clip(pnl_pct, -0.25, 0.25)) + 0.3 * float(np.clip(impact_pct * 4.0, -0.25, 0.25))
                new_adj = old_adj * LEARNING_DECAY + LEARNING_RATE * signal
            else:
                signal = float(current_signal.get(ticker, 0.0)) * LEARNING_IDLE_DECAY
                new_adj = old_adj * LEARNING_IDLE_DECAY

            new_adj = float(np.clip(new_adj, -LEARNING_CAP, LEARNING_CAP))
            rows.append((ticker, new_adj, float(signal), next_asof_date, updated_at))

        if rows:
            conn.executemany(
                """
                INSERT INTO learned_params (ticker, score_adjustment, last_signal, effective_asof, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    score_adjustment = excluded.score_adjustment,
                    last_signal = excluded.last_signal,
                    effective_asof = excluded.effective_asof,
                    updated_at = excluded.updated_at
                """,
                rows,
            )

        return {"updated": len(rows), "effective_asof": next_asof_date}


def sync_latest_realised_performance(current_asof_date: str) -> Optional[dict]:
    current_asof_date = _iso_date(current_asof_date)
    with get_conn() as conn:
        prev_asof_date = _latest_snapshot_before(conn, current_asof_date)
    if not prev_asof_date:
        return None

    perf = compute_month_performance(prev_asof_date, current_asof_date)
    if not perf:
        return None
    learning = update_learning_from_performance(prev_asof_date, current_asof_date)
    perf["learning_updated"] = learning["updated"]
    return perf


def get_learning_adjustments(asof_date: str) -> Dict[str, float]:
    asof_date = _iso_date(asof_date)
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT ticker, score_adjustment
            FROM learned_params
            WHERE effective_asof IS NULL OR effective_asof <= ?
            """,
            conn,
            params=(asof_date,),
        )
    if df.empty:
        return {}
    return {
        str(row["ticker"]).lower(): float(row["score_adjustment"])
        for _, row in df.iterrows()
        if abs(float(row["score_adjustment"])) > 1e-9
    }


def get_portfolio_history(limit: Optional[int] = None) -> List[dict]:
    with get_conn() as conn:
        latest_price_date = _latest_price_date(conn)
        where = ""
        params: List[object] = []
        if latest_price_date is not None:
            where = "WHERE s.asof_date <= ?"
            params.append(_iso_date(latest_price_date))

        sql = f"""
        SELECT
            s.asof_date,
            s.starting_equity,
            s.created_at,
            p.end_date,
            p.end_equity,
            p.pnl_gbp,
            p.pnl_pct,
            p.max_drawdown,
            p.spy_pnl_pct
        FROM portfolio_snapshots s
        LEFT JOIN portfolio_performance p
          ON p.asof_date = s.asof_date
        {where}
        ORDER BY s.asof_date DESC
        """
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return []
    return _records_json_safe(df)


def get_portfolio_performance(asof_date: Optional[str] = None) -> Optional[dict]:
    with get_conn() as conn:
        latest_price_date = _latest_price_date(conn)
        price_ceiling = _iso_date(latest_price_date) if latest_price_date is not None else None
        snapshot_lookup_date = _iso_date(asof_date) if asof_date is not None else None
        if snapshot_lookup_date is not None and price_ceiling is not None and snapshot_lookup_date > price_ceiling:
            return None

        if asof_date is None:
            if price_ceiling is not None:
                row = conn.execute(
                    """
                    SELECT asof_date
                    FROM portfolio_performance
                    WHERE end_date <= ?
                    ORDER BY end_date DESC, asof_date DESC
                    LIMIT 1
                    """,
                    (price_ceiling,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT asof_date
                    FROM portfolio_performance
                    ORDER BY end_date DESC, asof_date DESC
                    LIMIT 1
                    """
                ).fetchone()
            if not row:
                latest_snapshot_asof = _latest_snapshot_asof(conn, price_ceiling)
                if latest_snapshot_asof is None:
                    return None
                latest_snapshot = pd.read_sql_query(
                    """
                    SELECT asof_date, starting_equity
                    FROM portfolio_snapshots
                    WHERE asof_date = ?
                    """,
                    conn,
                    params=(latest_snapshot_asof,),
                )
                if latest_snapshot.empty:
                    return None
                snap = latest_snapshot.iloc[0]
                equity = float(snap["starting_equity"])
                asof = str(snap["asof_date"])
                return {
                    "asof_date": asof,
                    "end_date": asof,
                    "start_equity": equity,
                    "end_equity": equity,
                    "current_equity": equity,
                    "current_snapshot_date": asof,
                    "pnl_gbp": 0.0,
                    "pnl_pct": 0.0,
                    "mom_change_gbp": 0.0,
                    "mom_change_pct": 0.0,
                    "max_drawdown": 0.0,
                    "spy_start": None,
                    "spy_end": None,
                    "spy_pnl_pct": None,
                    "contributors": [],
                    "best_contributors": [],
                    "worst_contributors": [],
                    "curve": [{"date": asof, "equity": equity}],
                }
            asof_date = row[0]
        else:
            asof_date = snapshot_lookup_date

        perf = pd.read_sql_query(
            """
            SELECT *
            FROM portfolio_performance
            WHERE asof_date = ?
            """,
            conn,
            params=(asof_date,),
        )
        if perf.empty:
            if price_ceiling is not None and str(asof_date) > price_ceiling:
                return None
            latest_snapshot = pd.read_sql_query(
                """
                SELECT asof_date, starting_equity
                FROM portfolio_snapshots
                WHERE asof_date = ?
                """,
                conn,
                params=(asof_date,),
            )
            if latest_snapshot.empty:
                return None
            snap = latest_snapshot.iloc[0]
            equity = float(snap["starting_equity"])
            asof = str(snap["asof_date"])
            return {
                "asof_date": asof,
                "end_date": asof,
                "start_equity": equity,
                "end_equity": equity,
                "current_equity": equity,
                "current_snapshot_date": asof,
                "pnl_gbp": 0.0,
                "pnl_pct": 0.0,
                "mom_change_gbp": 0.0,
                "mom_change_pct": 0.0,
                "max_drawdown": 0.0,
                "spy_start": None,
                "spy_end": None,
                "spy_pnl_pct": None,
                "contributors": [],
                "best_contributors": [],
                "worst_contributors": [],
                "curve": [{"date": asof, "equity": equity}],
            }

        perf_row = perf.iloc[0].to_dict()
        contributors = pd.read_sql_query(
            """
            SELECT ticker, pnl_gbp, pnl_pct, contribution_pct
            FROM portfolio_contrib
            WHERE asof_date = ?
            ORDER BY pnl_gbp DESC, ticker ASC
            """,
            conn,
            params=(asof_date,),
        )
        curve = _build_daily_curve(conn, asof_date, perf_row["end_date"])
        current_snapshot_params: List[object] = [str(perf_row["end_date"])]
        current_snapshot_filter = "WHERE asof_date >= ?"
        if price_ceiling is not None:
            current_snapshot_filter += " AND asof_date <= ?"
            current_snapshot_params.append(price_ceiling)
        latest_snapshot = pd.read_sql_query(
            f"""
            SELECT asof_date, starting_equity
            FROM portfolio_snapshots
            {current_snapshot_filter}
            ORDER BY asof_date DESC
            LIMIT 1
            """,
            conn,
            params=current_snapshot_params,
        )

    contributors_list = _records_json_safe(contributors)
    best = _records_json_safe(contributors.sort_values("pnl_gbp", ascending=False).head(3))
    worst = _records_json_safe(contributors.sort_values("pnl_gbp", ascending=True).head(3))
    latest_snapshot_info = _json_safe_value(latest_snapshot.iloc[0].to_dict()) if not latest_snapshot.empty else None
    current_equity = None
    current_snapshot_date = None
    if latest_snapshot_info is not None:
        current_equity = float(latest_snapshot_info["starting_equity"])
        current_snapshot_date = str(latest_snapshot_info["asof_date"])

    return _json_safe_value({
        "asof_date": str(perf_row["asof_date"]),
        "end_date": str(perf_row["end_date"]),
        "start_equity": float(perf_row["start_equity"]),
        "end_equity": float(perf_row["end_equity"]),
        "current_equity": current_equity if current_equity is not None else float(perf_row["end_equity"]),
        "current_snapshot_date": current_snapshot_date or str(perf_row["end_date"]),
        "pnl_gbp": float(perf_row["pnl_gbp"]),
        "pnl_pct": float(perf_row["pnl_pct"]),
        "mom_change_gbp": float(perf_row["pnl_gbp"]),
        "mom_change_pct": float(perf_row["pnl_pct"]),
        "max_drawdown": float(perf_row["max_drawdown"]),
        "spy_start": float(perf_row["spy_start"]) if perf_row["spy_start"] is not None else None,
        "spy_end": float(perf_row["spy_end"]) if perf_row["spy_end"] is not None else None,
        "spy_pnl_pct": float(perf_row["spy_pnl_pct"]) if perf_row["spy_pnl_pct"] is not None else None,
        "contributors": contributors_list,
        "best_contributors": best,
        "worst_contributors": worst,
        "curve": [
            {"date": str(pd.to_datetime(row["date"]).date()), "equity": float(row["equity"])}
            for _, row in curve.iterrows()
        ],
    })


def get_current_portfolio_status(snapshot_asof: Optional[str] = None) -> Optional[dict]:
    with get_conn() as conn:
        latest_price_date = _latest_price_date(conn)
        if latest_price_date is None:
            return None
        latest_price_date = _iso_date(latest_price_date)

        if snapshot_asof is None:
            snapshot_asof = _latest_snapshot_asof(conn, latest_price_date)
        if snapshot_asof is None:
            return None
        snapshot_asof = _iso_date(snapshot_asof)
        if snapshot_asof > latest_price_date:
            return None

        holdings = _load_snapshot_holdings(conn, snapshot_asof)
        if holdings.empty:
            return None

        end_price_map = _price_map_on_or_before(conn, latest_price_date, holdings["ticker"].tolist())
        holdings["end_price"] = holdings["ticker"].map(end_price_map).fillna(0.0).astype(float)
        holdings.loc[holdings["ticker"] == "cash", "end_price"] = 1.0

        holdings["start_value"] = holdings["shares"] * holdings["price"]
        holdings["current_value"] = holdings["shares"] * holdings["end_price"]
        holdings["pnl_gbp"] = holdings["current_value"] - holdings["start_value"]
        holdings["pnl_pct"] = np.where(
            holdings["start_value"].abs() > 1e-12,
            holdings["current_value"] / holdings["start_value"] - 1.0,
            0.0,
        )

        start_equity = float(holdings["start_value"].sum())
        current_equity = float(holdings["current_value"].sum())
        pnl_gbp = float(current_equity - start_equity)
        pnl_pct = float(current_equity / start_equity - 1.0) if abs(start_equity) > 1e-12 else 0.0

        if abs(pnl_gbp) > 1e-12:
            holdings["contribution_pct"] = holdings["pnl_gbp"] / pnl_gbp
        else:
            holdings["contribution_pct"] = 0.0

        curve = _build_daily_curve(conn, snapshot_asof, latest_price_date)
        if curve.empty:
            max_drawdown = 0.0
        else:
            curve["running_max"] = curve["equity"].cummax()
            curve["drawdown"] = curve["equity"] / curve["running_max"] - 1.0
            max_drawdown = float(curve["drawdown"].min())

        spy_start = _single_close_on_or_before(conn, "spy.us", snapshot_asof)
        spy_end = _single_close_on_or_before(conn, "spy.us", latest_price_date)
        spy_pnl_pct = None
        if spy_start and spy_end and spy_start > 0:
            spy_pnl_pct = float(spy_end / spy_start - 1.0)

    contributors = holdings[["ticker", "pnl_gbp", "pnl_pct", "contribution_pct"]].sort_values(
        ["pnl_gbp", "ticker"], ascending=[False, True]
    )
    best = contributors.head(5)
    worst = contributors.sort_values(["pnl_gbp", "ticker"], ascending=[True, True]).head(5)

    return _json_safe_value(
        {
            "snapshot_asof_date": snapshot_asof,
            "price_date": latest_price_date,
            "start_equity": start_equity,
            "current_equity": current_equity,
            "pnl_gbp": pnl_gbp,
            "pnl_pct": pnl_pct,
            "max_drawdown": float(max_drawdown),
            "spy_start": float(spy_start) if spy_start is not None else None,
            "spy_end": float(spy_end) if spy_end is not None else None,
            "spy_pnl_pct": float(spy_pnl_pct) if spy_pnl_pct is not None else None,
            "contributors": _records_json_safe(contributors),
            "best_contributors": _records_json_safe(best),
            "worst_contributors": _records_json_safe(worst),
            "curve": [
                {"date": str(pd.to_datetime(row["date"]).date()), "equity": float(row["equity"])}
                for _, row in curve.iterrows()
            ],
        }
    )
