import argparse

from src.pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the monthly investing pipeline.")
    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=None,
        help="Actual Trading212 account value to use for trade sizing and the new snapshot.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_pipeline(portfolio_value=args.portfolio_value)
