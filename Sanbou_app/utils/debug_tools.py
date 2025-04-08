"""
デバッグ用ユーティリティ関数
"""

import os
import pandas as pd
from typing import Dict


def save_debug_csvs(dfs: Dict[str, pd.DataFrame], folder: str = "debug_data") -> None:
    os.makedirs(folder, exist_ok=True)
    for name, df in dfs.items():
        df.to_csv(os.path.join(folder, f"debug_{name}.csv"), index=False)


if __name__ == "__main__":
    # テスト用コード
    df = pd.DataFrame({"日付": ["2024-01-01"], "数量": [100]})
    save_debug_csvs({"テスト": df})
