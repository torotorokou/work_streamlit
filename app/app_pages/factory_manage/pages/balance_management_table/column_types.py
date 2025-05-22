import pandas as pd
from utils.config_loader import get_expected_dtypes_by_template


def load_column_types(schema_name: str = "balance_management_table") -> dict:
    """
    YAMLファイルからカラムの型定義を読み込む

    Args:
        schema_name (str): 使用するスキーマ名（デフォルトは'balance_management_table'）

    Returns:
        dict: カラム名とデータ型の対応辞書
    """
    return get_expected_dtypes_by_template(schema_name).get("shipping", {})


# カラムの型定義を読み込む
COLUMN_TYPES = load_column_types()


def apply_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームに定義されたデータ型を適用する関数

    Args:
        df (pd.DataFrame): 入力データフレーム

    Returns:
        pd.DataFrame: データ型が適用されたデータフレーム
    """
    # 日付型の変換を個別に処理
    if "伝票日付" in df.columns:
        df["伝票日付"] = pd.to_datetime(
            df["伝票日付"], format="%Y-%m-%d", errors="coerce"
        )

    # 数値型のカラムは欠損値をNaNに変換してから型変換
    numeric_columns = ["正味重量", "数量", "単価", "金額"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # その他のカラムの型変換
    for column, dtype in COLUMN_TYPES.items():
        if column in df.columns and column not in ["伝票日付"] + numeric_columns:
            df[column] = df[column].astype(dtype)

    return df
