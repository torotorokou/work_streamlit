import pandas as pd
import numpy as np


def clean_numeric_column(df, column_name):
    """
    指定された列を float に変換（カンマや空文字を考慮）
    """
    df[column_name] = (
        df[column_name]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", pd.NA)
        .astype(float)
    )
    return df


def enforce_dtypes(df, dtype_map):
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                if dtype == "datetime64[ns]":
                    # 括弧つき曜日などを除去して日付変換
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(r"\(.+?\)", "", regex=True)
                        .str.strip()
                    )
                    df[col] = pd.to_datetime(df[col], errors="coerce")

                elif dtype in [int, np.int64]:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)  # カンマ削除を追加
                        .str.strip()
                        .replace("", pd.NA)
                    )
                    df[col] = (
                        pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                    )

                elif dtype in [float, np.float64]:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                        .str.strip()
                        .replace("", pd.NA)
                        .astype(float)
                    )

                else:
                    df[col] = df[col].astype(dtype)

            except Exception as e:
                print(f"⚠️ {col} の型変換に失敗: {e}")
    return df
