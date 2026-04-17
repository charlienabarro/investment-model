# Changes

Implemented the Claude review improvement tiers across the main model, optimiser,
Commodity Scout, features, backtest, and API.

## Modified files

- `src/model.py`
  - Carries actual prior risky-sleeve weights between monthly optimiser runs.
  - Scales model scores by `0.04` before MVO so dimensionless ranks behave like
    expected-return proxies.
  - Computes `fwd_ret_1m` inside per-slice training/evaluation windows instead
    of on the full month-end frame.
  - Adds a one-month validation embargo.
  - Keeps learning overlay modest and gated to clearly strong current scores.
  - Adds decomposed recommendation reasons: ML signal, news adjustment,
    learning adjustment, regime, account weight, and pie weight.
  - Adds a hard drawdown circuit breaker for cash.
  - Adds market regime detection and regime-aware budget tilts.
  - Adds `high_52w_ratio_z` and `vol_of_vol_21_z` to ML/fallback scoring.
  - Adds optional Ridge as a third ensemble member.

- `src/optimiser.py`
  - Uses equal-weight fallback when solver failure occurs with all-zero prior
    weights, avoiding NaN fallback portfolios.

- `src/commodity_scout.py`
  - Adds a 42-day embargo to commodity LightGBM training and requires at least
    180 uncontaminated rows.
  - Replaces DataFrame `.get(..., scalar)` patterns with safe Series defaults.
  - Adds commodity-specific news event impact weights.
  - Adds inverse-volatility position sizing.
  - Adds roll-yield proxy features for spot/ETF versus futures pairs.
  - Adds optional FinBERT headline sentiment before VADER/keyword fallback.

- `src/portfolio_tracking.py`
  - Documents and keeps the lower learning rate so performance feedback does
    not double-count momentum.

- `src/features.py`
  - Adds `high_52w_ratio` and `vol_of_vol_21` feature engineering.
  - Adds both new features to the cross-sectional z-score pipeline.

- `src/backtest.py`
  - Returns a 60/40 SPY/IEF benchmark series alongside strategy curve and stats.

- `src/api.py`
  - Adds `benchmark_6040` to `/backtest/equity`.

- `src/risk_policy.py`
  - Documents commonly unmapped tickers for future sector mapping.
  - The main model now applies a tighter explicit `"Other"` sector cap.

- `src/news_sentiment.py`
  - Adds optional FinBERT headline sentiment before VADER fallback.

## TODO comments left

- `src/commodity_scout.py`
  - FinBERT first run can require a large model download; consider configuring
    `HF_HOME` or a local `data/finbert/` cache.
  - Roll-yield currently uses ETF/continuous-future proxies; replace with true
    futures curve data when a reliable free source is available.

- `src/news_sentiment.py`
  - Same FinBERT cache note as above.

- `src/risk_policy.py`
  - Add sector mappings for common unmapped universe names such as `tsla.us`,
    `brk.b.us`, `cost.us`, `nflx.us`, `avgo.us`, `lly.us`, `unh.us`, `tmo.us`,
    `dhr.us`, `lin.us`, `now.us`, `pld.us`, `isrg.us`, and `eqix.us`.

## Verification

- `python test.py` without arguments is not runnable in this repo because it
  requires `--start`.
- `python test.py --start 2005-01-01` was started but stayed silent for several
  minutes, so it was stopped and replaced with shorter tier gates.
- Tier gates run successfully:
  - `/opt/anaconda3/bin/python test.py --start 2024-01-01 --mode db --out /tmp/investing_tier1_equity.csv`
  - `/opt/anaconda3/bin/python test.py --start 2024-01-01 --mode db --out /tmp/investing_tier2_equity.csv`
  - `/opt/anaconda3/bin/python test.py --start 2024-01-01 --mode db --out /tmp/investing_tier3_equity.csv`
- Compile check passed:
  - `/opt/anaconda3/bin/python -m py_compile src/model.py src/commodity_scout.py src/features.py src/news_sentiment.py src/backtest.py src/api.py`
- Full main pipeline completed:
  - `/opt/anaconda3/bin/python run_pipeline.py --portfolio-value 10000`
  - External price/news fetches emitted connection warnings in the sandbox, but
    DB-backed recommendation generation, holdings/trades, and snapshot saving
    completed.
- Commodity pipeline completed:
  - `/opt/anaconda3/bin/python run_commodity_pipeline.py --portfolio-value 10000`
  - External price/news fetches emitted connection warnings in the sandbox, but
    Commodity Scout generated recommendations, holdings, and trades.
- API startup check passed after allowing local bind:
  - `/opt/anaconda3/bin/python run_api.py`
- Latest DB allocation checks after pipeline:
  - Main recommendations sum to `1.0`.
  - Commodity recommendations sum to `1.0`.
  - Commodity invested pie weights excluding cash sum to `1.0`.
  - Commodity non-cash weights are within the `10%` to `35%` rule.
