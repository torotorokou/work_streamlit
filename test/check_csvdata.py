import sqlite3
import pandas as pd
from utils.config_loader import get_path_from_yaml

# --- SQLite DBパスとテーブル名 ---
db_path = get_path_from_yaml("weight_data", section="sql_database")
table_name = "ukeire"

# --- SQLiteに接続して読み込む ---
conn = sqlite3.connect(db_path)

# --- 目的の日付でフィルタリング（文字列比較） ---
query = f"""
SELECT * FROM {table_name}
WHERE 伝票日付 = '2025-04-20'
"""

df = pd.read_sql(query, conn)

conn.close()

# --- 結果を表示 ---
print(df)
