import pandas as pd
import numpy as np
from utils.logger import app_logger
from components.ui_message import show_warning


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
    logger = app_logger()
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
                msg = f"⚠️ {col} の型変換に失敗: {e}"
                logger.warning(msg)
                show_warning(msg)

    return df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame内の文字列列に対して、
    ・前後の空白（半角/全角）
    ・中央の空白（半角/全角）
    をすべて削除する。

    Parameters:
        df (pd.DataFrame): 処理対象のデータフレーム

    Returns:
        pd.DataFrame: 空白が除去された新しいデータフレーム
    """
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()  # 前後の半角スペース, 改行, タブなどを除去
            .str.replace("　", "", regex=False)  # 全角スペースをすべて削除
            .str.replace(" ", "", regex=False)  # 半角スペースをすべて削除
        )
    return df
