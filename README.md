# investment-model

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Current State
- Latest valid price date: 2026-04-15.
- Latest valid portfolio snapshot: 2026-04-15.
- Current portfolio value is inferred from the latest saved snapshot marked to current prices.
- Future-dated generated rows are ignored by the app and were cleaned from the local database after backup.

## Monthly Task
Run this when you want new recommendations and a new trade plan. By default it refreshes prices, marks the current saved snapshot to market, and uses that inferred equity for trade sizing and the new snapshot.

```bash
python run_pipeline.py
```

If you need to force a manual account value, use:

```bash
python run_pipeline.py --portfolio-value <actual-account-value>
```

## Weekly Task
Run this between monthly recommendation runs. It refreshes prices and marks the current saved portfolio to market without generating new trades.

```bash
python run_weekly_check.py
```

If prices are already up to date and you only want to inspect the stored state:

```bash
python run_weekly_check.py --skip-price-update
```

## Dashboard
Run the API continuously with:

```bash
python run_api.py --serve
```

For a quick startup smoke check only:

```bash
python run_api.py
```

## Commodity Scout
Commodity Scout is a separate commodities-only Trading 212 pie. It does not change the main portfolio recommendations.

Run the commodity pipeline with live commodity prices and Google News RSS:

```bash
python run_commodity_pipeline.py
```

Useful options:

```bash
python run_commodity_pipeline.py --portfolio-value <commodity-account-value>
python run_commodity_pipeline.py --skip-prices --skip-news
```

Dashboard endpoints:
- `/commodities/latest` shows the latest commodity target allocation including cash outside the pie.
- `/commodities/export/trading212.csv` exports the invested Trading 212 pie weights, excluding cash and summing to 100%.
- `/commodities/backtest` returns the Commodity Scout backtest versus DBC.
