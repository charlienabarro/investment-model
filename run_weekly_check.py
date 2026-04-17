from __future__ import annotations

import argparse
import json
import sys

from src.db import init_db
from src.portfolio_tracking import get_current_portfolio_status


def _fmt_pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_gbp(value) -> str:
    if value is None:
        return "n/a"
    return f"GBP {float(value):,.2f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh prices and mark the latest saved portfolio snapshot to market without generating new recommendations."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the weekly portfolio status as JSON.",
    )
    parser.add_argument(
        "--skip-price-update",
        action="store_true",
        help="Use existing DB prices without fetching fresh market data.",
    )
    return parser


def _print_human_report(status: dict) -> None:
    print(
        f"[OK] Weekly portfolio check from {status['snapshot_asof_date']} "
        f"to latest price date {status['price_date']}"
    )
    print(f"[OK] Current equity: {_fmt_gbp(status['current_equity'])}")
    print(
        f"[OK] Gain/loss since snapshot: {_fmt_gbp(status['pnl_gbp'])} "
        f"({_fmt_pct(status['pnl_pct'])})"
    )
    print(f"[OK] SPY same-period return: {_fmt_pct(status['spy_pnl_pct'])}")
    print(f"[OK] Max drawdown since snapshot: {_fmt_pct(status['max_drawdown'])}")

    best = status.get("best_contributors") or []
    worst = status.get("worst_contributors") or []
    if best:
        top = best[0]
        print(
            f"[OK] Best contributor: {top['ticker']} "
            f"{_fmt_gbp(top['pnl_gbp'])} ({_fmt_pct(top['pnl_pct'])})"
        )
    if worst:
        bottom = worst[0]
        print(
            f"[OK] Worst contributor: {bottom['ticker']} "
            f"{_fmt_gbp(bottom['pnl_gbp'])} ({_fmt_pct(bottom['pnl_pct'])})"
        )


def main() -> int:
    args = _build_parser().parse_args()

    init_db()
    if not args.skip_price_update:
        from src.pipeline import update_all_prices

        print("[INFO] Refreshing daily prices only. Recommendations and trades are unchanged.")
        update_all_prices()

    status = get_current_portfolio_status()
    if status is None:
        print(
            "[ERROR] No saved portfolio snapshot found. "
            "Run python run_pipeline.py first to create the monthly portfolio.",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        _print_human_report(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
