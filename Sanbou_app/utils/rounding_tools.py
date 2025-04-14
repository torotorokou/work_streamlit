import pandas as pd
import numpy as np


def round_value_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    「小項目1」または「小項目2」に「単価」が含まれる行は小数点第2位に丸め、
    それ以外は整数に丸める。
    ただし、丸めた結果が0.00なら 0 に変換する。
    """
    is_tanka = df["小項目1"].astype(str).str.contains("単価", na=False) | df[
        "小項目2"
    ].astype(str).str.contains("単価", na=False)

    df["値"] = pd.to_numeric(df["値"], errors="coerce")

    # 丸め処理
    rounded = np.where(is_tanka, df["値"].round(2), df["値"].round(0))

    # 値が0.00なら0に置き換え（型は float → int にせず float のまま 0.0 → 0）
    df["値"] = np.where(rounded == 0, 0, rounded)

    return df
