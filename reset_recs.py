from src.db import get_conn, init_db

init_db()
with get_conn() as conn:
    conn.execute("DELETE FROM recommendations;")
    conn.execute("DELETE FROM model_holdings;")
    conn.execute("DELETE FROM model_trades;")

print("Cleared recommendations, holdings, trades.")