# src/api.py — V3 with improved dashboard
import json
import io
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from .db import get_conn, init_db
from .universe import get_universe
from .backtest import run_recommendation_backtest

app = FastAPI(title="Investing Backend")


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
        cur = conn.execute("SELECT MAX(asof_date) FROM recommendations")
        row = cur.fetchone()
        if not row or row[0] is None:
            return {"asof_date": None, "recs": []}
        asof = row[0]
        df = pd.read_sql_query(
            """
            SELECT asof_date, ticker, action, score, target_weight, reasons
            FROM recommendations
            WHERE asof_date = ? AND action = 'BUY_OR_HOLD' AND target_weight >= 0.03
            ORDER BY target_weight DESC, score DESC, ticker ASC
            """,
            conn,
            params=(asof,),
        )
    return {"asof_date": asof, "recs": df.to_dict(orient="records")}


@app.get("/recommendations/history")
def recommendations_history(limit: int = Query(2, ge=1, le=24)):
    with get_conn() as conn:
        dates = pd.read_sql_query(
            "SELECT DISTINCT asof_date FROM recommendations ORDER BY asof_date DESC LIMIT ?",
            conn,
            params=(limit,),
        )
        out = []
        for asof in dates["asof_date"].tolist():
            df = pd.read_sql_query(
                """
                SELECT asof_date, ticker, action, score, target_weight, reasons
                FROM recommendations
                WHERE asof_date = ? AND action = 'BUY_OR_HOLD' AND target_weight >= 0.03
                ORDER BY target_weight DESC, score DESC, ticker ASC
                """,
                conn,
                params=(asof,),
            )
            out.append({"asof_date": asof, "recs": df.to_dict(orient="records")})
    return {"snapshots": out}


def _load_prices_and_recs():
    universe = get_universe()
    placeholders = ",".join(["?"] * len(universe))
    with get_conn() as conn:
        prices = pd.read_sql_query(
            f"SELECT date, ticker, close FROM prices_daily WHERE ticker IN ({placeholders}) ORDER BY date ASC",
            conn,
            params=universe,
        )
        recs = pd.read_sql_query(
            "SELECT asof_date, ticker, target_weight FROM recommendations ORDER BY asof_date ASC",
            conn,
        )
    return prices, recs


@app.get("/backtest/equity")
def backtest_equity(cost_bps: float = 5.0, max_points: int = 2000):
    prices, recs = _load_prices_and_recs()
    curve, stats = run_recommendation_backtest(prices, recs, cost_bps=cost_bps)
    if curve.empty:
        return {"stats": stats, "series": []}
    if len(curve) > max_points:
        curve = curve.iloc[:: max(1, len(curve) // max_points)].copy()
    series = [
        {"date": str(d.date()), "equity": float(e)}
        for d, e in zip(pd.to_datetime(curve["date"]), curve["equity"])
    ]
    return {"stats": stats, "series": series}


@app.get("/export/rebalance_pack.csv")
def export_rebalance_pack_csv(portfolio_value: float = Query(350.0, gt=0)):
    with get_conn() as conn:
        dates = pd.read_sql_query(
            "SELECT DISTINCT asof_date FROM recommendations ORDER BY asof_date DESC LIMIT 2",
            conn,
        )
        if dates.empty:
            return Response(content="message\nNo recommendations found\n", media_type="text/csv")

        latest_asof = dates["asof_date"].iloc[0]
        prev_asof = dates["asof_date"].iloc[1] if len(dates) > 1 else None

        latest = pd.read_sql_query(
            "SELECT ticker, target_weight, action, score, reasons FROM recommendations WHERE asof_date = ? AND action = 'BUY_OR_HOLD' AND target_weight >= 0.03",
            conn,
            params=(latest_asof,),
        )
        prev = pd.DataFrame(columns=["ticker", "target_weight"])
        if prev_asof:
            prev = pd.read_sql_query(
                "SELECT ticker, target_weight FROM recommendations WHERE asof_date = ? AND action = 'BUY_OR_HOLD' AND target_weight >= 0.03",
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

    latest_w = latest[["ticker", "target_weight"]].rename(columns={"target_weight": "new_weight"})
    prev_w = prev[["ticker", "target_weight"]].rename(columns={"target_weight": "prev_weight"}) if not prev.empty else pd.DataFrame(columns=["ticker", "prev_weight"])

    pack = latest_w.merge(prev_w, on="ticker", how="outer").fillna(0.0)
    pack = pack.merge(latest[["ticker", "action", "score", "reasons"]], on="ticker", how="left")
    pack = pack.merge(prices, on="ticker", how="left")
    pack["delta_weight"] = pack["new_weight"] - pack["prev_weight"]

    def classify(row):
        nw, pw = float(row["new_weight"]), float(row["prev_weight"])
        if pw == 0 and nw > 0: return "NEW_BUY"
        if pw > 0 and nw == 0: return "EXIT_SELL"
        if nw > pw: return "BUY"
        if nw < pw: return "SELL"
        return "HOLD"

    pack["rebalance_action"] = pack.apply(classify, axis=1)
    pack["portfolio_value"] = float(portfolio_value)
    pack["est_shares_delta"] = pack.apply(
        lambda r: 0.0 if pd.isna(r["price"]) or float(r["price"]) == 0.0
        else (float(r["delta_weight"]) * float(portfolio_value)) / float(r["price"]),
        axis=1,
    )
    pack["est_notional"] = pack["est_shares_delta"].abs() * pack["price"].fillna(0.0)

    action_order = {"NEW_BUY": 0, "BUY": 1, "SELL": 2, "EXIT_SELL": 3, "HOLD": 4}
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
      <span class="pill">Portfolio <b id="pvLabel">350</b></span>
      <input id="portfolioValue" type="number" value="350" min="1" step="50">
      <button onclick="reloadAll()">Refresh</button>
    </div>
  </div>

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
          <button onclick="downloadPieCSV()">Download T212 Pie CSV</button>
          <button onclick="downloadRebalancePack()">Download Full CSV</button>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">Run <code>python run_pipeline.py</code> monthly. Keep <code>python run_api.py</code> running for this UI.</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<script>
// ═══════════════════════════════════════════════════════
// Ticker → friendly name map
// ═══════════════════════════════════════════════════════
const NAMES = {
  // ETFs
  "spy.us":"S&P 500 Index","qqq.us":"Nasdaq 100 Index","dia.us":"Dow Jones Index","iwm.us":"Russell 2000 Small Cap",
  "vti.us":"Total US Stock Market","voo.us":"S&P 500 (Vanguard)","ief.us":"7-10 Year US Treasuries","tlt.us":"20+ Year US Treasuries",
  "shy.us":"Short-Term Treasuries","lqd.us":"Investment Grade Bonds","hyg.us":"High Yield (Junk) Bonds",
  "gld.us":"Gold","iau.us":"Gold (iShares)","slv.us":"Silver",
  "dbc.us":"Broad Commodities","pdbc.us":"Broad Commodities (Invesco)","djp.us":"Commodity Index",
  "uso.us":"Oil","ung.us":"Natural Gas","dba.us":"Agriculture","cper.us":"Copper","copx.us":"Copper Miners",
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
function getTicker(t) { return t.replace(/\.us$/i,"").toUpperCase(); }

function bucketLabel(t) {
  t = t.toLowerCase();
  const bonds = ["ief.us","tlt.us","shy.us","lqd.us","hyg.us"];
  const comms = ["gld.us","iau.us","slv.us","dbc.us","pdbc.us","djp.us","uso.us","ung.us","dba.us","cper.us","copx.us"];
  if (bonds.includes(t)) return "Bond ETF";
  if (comms.includes(t)) return "Commodity";
  const etfs = ["spy.us","qqq.us","dia.us","iwm.us","vti.us","voo.us"];
  if (etfs.includes(t)) return "Index ETF";
  return "Stock";
}

// ═══════════════════════════════════════════════════════
// Globals
// ═══════════════════════════════════════════════════════
let equityChart = null, latestRecs = null, prevRecs = null, latestPrices = null;
let allSeries = [], currentTimeframe = "ALL";

const fmtPct = x => x == null ? "n/a" : (x*100).toFixed(1)+"%";
const fmtGBP = x => x == null ? "n/a" : "\u00a3"+Number(x).toFixed(2);
const fmtNum = (x,d=2) => x == null ? "n/a" : Number(x).toFixed(d);
async function fetchJson(u) { return (await fetch(u)).json(); }
function priceMap() { const m={}; (latestPrices||[]).forEach(p=>m[p.ticker]=Number(p.close)); return m; }
function getPV() { const v=Number(document.getElementById("portfolioValue").value||0); return v>0?v:350; }
function downloadRebalancePack() { window.location="/export/rebalance_pack.csv?portfolio_value="+getPV(); }

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
  if (bucket === "Bond ETF") {
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

// ═══════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════
function renderSummary() {
  const el = document.getElementById("summaryText");
  if (!latestRecs || !latestRecs.recs || !latestRecs.recs.length) { el.textContent = "No recommendations yet. Run the pipeline first."; return; }
  const recs = latestRecs.recs.filter(r => Number(r.target_weight) >= 0.03);
  const pv = getPV();
  let nE=0,nB=0,nC=0;
  recs.forEach(r => { const b=bucketLabel(r.ticker); if(b==="Bond ETF")nB++; else if(b==="Commodity")nC++; else nE++; });
  const big = recs[0], bigPct = (Number(big.target_weight)*100).toFixed(0);

  let s = `<b>As of ${latestRecs.asof_date}</b> \u2014 Your \u00a3${pv} portfolio holds <b>${recs.length} positions</b>: ${nE} stock${nE!==1?"s":""}, ${nB} bond${nB!==1?"s":""}, and ${nC} commodit${nC!==1?"ies":"y"}. `;
  s += `The biggest position is <b>${getName(big.ticker)}</b> at ${bigPct}% (\u00a3${(Number(big.target_weight)*pv).toFixed(0)}).`;
  el.innerHTML = s;
}

// ═══════════════════════════════════════════════════════
// Holdings cards
// ═══════════════════════════════════════════════════════
function renderHoldings() {
  const c = document.getElementById("holdingsCards"); c.innerHTML = "";
  if (!latestRecs || !latestRecs.recs) return;
  const recs = latestRecs.recs.filter(r => Number(r.target_weight) >= 0.03);
  const pv = getPV(), pm = priceMap();
  document.getElementById("holdingsExplainer").textContent = `${recs.length} positions ordered by size. Tap any for details.`;

  recs.forEach(r => {
    const w=Number(r.target_weight), px=pm[r.ticker], val=w*pv, sh=px?val/px:0;
    const reason = humaniseReasons(r.ticker, r.reasons, w);
    const card = document.createElement("div"); card.className = "holding-card";
    card.innerHTML = `
      <div class="hc-top">
        <div><span class="hc-name">${getTicker(r.ticker)}</span> <span class="hc-full">${getName(r.ticker)} \u00b7 ${bucketLabel(r.ticker)}</span></div>
        <span class="hc-weight">${(w*100).toFixed(1)}%</span>
      </div>
      <div class="hc-bar"><div class="hc-bar-fill" style="width:${Math.min(w/0.20*100,100).toFixed(0)}%"></div></div>
      <div class="hc-details">${fmtGBP(val)}${px?" \u2014 ~"+fmtNum(sh)+" shares at "+fmtGBP(px):""}</div>
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
  const latest={}, prev={};
  latestRecs.recs.filter(r=>Number(r.target_weight)>=0.03).forEach(r=>latest[r.ticker]=Number(r.target_weight));
  if (prevRecs && prevRecs.recs) prevRecs.recs.filter(r=>Number(r.target_weight)>=0.03).forEach(r=>prev[r.ticker]=Number(r.target_weight));
  const changes=[];
  new Set([...Object.keys(latest),...Object.keys(prev)]).forEach(t=>{
    const wN=latest[t]||0, wO=prev[t]||0;
    if(wN>0&&wO===0) changes.push({ticker:t,type:"new",label:"New",desc:`Added at ${(wN*100).toFixed(0)}%`});
    else if(wN===0&&wO>0) changes.push({ticker:t,type:"exit",label:"Removed",desc:`Was ${(wO*100).toFixed(0)}%`});
    else if(Math.abs(wN-wO)>=0.02){const dir=wN>wO?"buy":"sell",verb=wN>wO?"Increased":"Reduced";changes.push({ticker:t,type:dir,label:verb,desc:`${(wO*100).toFixed(0)}% \u2192 ${(wN*100).toFixed(0)}%`});}
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
  const pm=priceMap(),pv=getPV(),latest={},prev={};
  latestRecs.recs.filter(r=>Number(r.target_weight)>=0.03).forEach(r=>latest[r.ticker]=Number(r.target_weight));
  if(prevRecs&&prevRecs.recs)prevRecs.recs.filter(r=>Number(r.target_weight)>=0.03).forEach(r=>prev[r.ticker]=Number(r.target_weight));
  const trades=[];
  new Set([...Object.keys(latest),...Object.keys(prev)]).forEach(t=>{
    const wN=latest[t]||0,wO=prev[t]||0,wD=wN-wO;if(Math.abs(wD)<0.02)return;
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
// Downloads
// ═══════════════════════════════════════════════════════
function downloadPieCSV() {
  if(!latestRecs||!latestRecs.recs)return;
  const rows=latestRecs.recs.filter(r=>Number(r.target_weight)>=0.03);
  let csv="Ticker,Weight,Percent\n";
  rows.forEach(r=>{const w=Number(r.target_weight);csv+=`${r.ticker},${w.toFixed(6)},${(w*100).toFixed(2)}\n`;});
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download=`trading212_pie_${latestRecs.asof_date||"latest"}.csv`;document.body.appendChild(a);a.click();document.body.removeChild(a);
}

// ═══════════════════════════════════════════════════════
// Data loading
// ═══════════════════════════════════════════════════════
async function loadPrices(){latestPrices=await fetchJson("/prices/latest");}
async function loadRecs(){latestRecs=await fetchJson("/recommendations/latest");}
async function loadRecHistory(){
  const d=await fetchJson("/recommendations/history?limit=2");const s=d.snapshots||[];
  if(s[0])latestRecs={asof_date:s[0].asof_date,recs:s[0].recs};
  prevRecs=s[1]?{asof_date:s[1].asof_date,recs:s[1].recs}:null;
}
function renderAll(){
  document.getElementById("pvLabel").textContent=String(getPV());
  renderSummary();renderHoldings();renderChanges();renderActions();renderMarketBox();
}
async function reloadAll(){
  await Promise.all([loadPrices(),loadRecs(),loadRecHistory()]);
  renderAll();await loadBacktest();
}
document.getElementById("portfolioValue").addEventListener("input",()=>renderAll());
reloadAll();
</script>
</body>
</html>
    """
    return HTMLResponse(content=html)


# ── Keep these for programmatic access ──

@app.get("/portfolio/holdings/latest")
def latest_holdings():
    with get_conn() as conn:
        cur = conn.execute("SELECT MAX(asof_date) FROM model_holdings")
        row = cur.fetchone()
        if not row or row[0] is None:
            return {"asof_date": None, "holdings": []}
        asof = row[0]
        df = pd.read_sql_query(
            "SELECT * FROM model_holdings WHERE asof_date = ? ORDER BY value DESC",
            conn, params=(asof,),
        )
    return {"asof_date": asof, "holdings": df.to_dict(orient="records")}


@app.get("/portfolio/trades/latest")
def latest_trades():
    with get_conn() as conn:
        cur = conn.execute("SELECT MAX(asof_date) FROM model_trades")
        row = cur.fetchone()
        if not row or row[0] is None:
            return {"asof_date": None, "trades": []}
        asof = row[0]
        df = pd.read_sql_query(
            "SELECT * FROM model_trades WHERE asof_date = ? ORDER BY est_notional DESC",
            conn, params=(asof,),
        )
    return {"asof_date": asof, "trades": df.to_dict(orient="records")}

