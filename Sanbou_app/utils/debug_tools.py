"""
デバッグ用ユーティリティ関数
"""

import os
import pandas as pd
from typing import Dict


def save_debug_csvs(dfs: Dict[str, pd.DataFrame], folder: str = "debug_data") -> None:
    os.makedirs(folder, exist_ok=True)
    for name, df in dfs.items():
        df.to_csv(os.path.join(folder, f"debug_{name}.csv"), index=False, encoding="utf-8-sig")


def check_dfs(dfs: dict, rows: int = 5, show_columns: bool = True):
    for key, df in dfs.items():
        print(f"\n📘 {key} - {len(df)}件")
        if show_columns:
            print("🧾 カラム:", df.columns.tolist())
        print(df.head(rows))