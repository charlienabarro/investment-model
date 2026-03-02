import io
import pandas as pd
import requests
import certifi

def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"

    r = requests.get(url, timeout=30, verify=certifi.where())
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))

    if df.empty:
        return df

    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["date", "close"])
    return df.sort_values("date").reset_index(drop=True)