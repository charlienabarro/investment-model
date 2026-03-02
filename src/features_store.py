import pandas as pd
from .db import get_conn, init_db

FEATURE_COLS = [
    "ticker", "date",
    "mom_12_1", "mom_6_1", "vol_63", "ma_200_ratio", "maxdd_252"
]

def upsert_features(df: pd.DataFrame) -> int:
    init_db()

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date.astype(str)

    keep = [c for c in FEATURE_COLS if c in d.columns]
    d = d[keep].copy()

    rows = d.to_records(index=False)

    cols_sql = ", ".join(keep)
    placeholders = ", ".join(["?"] * len(keep))

    # SQLite UPSERT
    update_sql = ", ".join([f"{c}=excluded.{c}" for c in keep if c not in ("ticker", "date")])

    sql = f"""
    INSERT INTO features_daily ({cols_sql})
    VALUES ({placeholders})
    ON CONFLICT(ticker, date) DO UPDATE SET
    {update_sql}
    """

    with get_conn() as conn:
        conn.executemany(sql, rows)
        return len(d)