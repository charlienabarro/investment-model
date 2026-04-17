from __future__ import annotations

import argparse

from src.commodity_scout import run_commodity_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Commodity Scout commodities-only pipeline.")
    parser.add_argument("--portfolio-value", type=float, default=None, help="Optional portfolio value for the commodity pie in GBP.")
    parser.add_argument("--skip-prices", action="store_true", help="Use existing price data instead of refreshing commodities.")
    parser.add_argument("--skip-news", action="store_true", help="Use existing/fallback news features instead of fetching Google News RSS.")
    args = parser.parse_args()
    run_commodity_pipeline(
        portfolio_value=args.portfolio_value,
        refresh_prices=not args.skip_prices,
        refresh_news=not args.skip_news,
    )


if __name__ == "__main__":
    main()
