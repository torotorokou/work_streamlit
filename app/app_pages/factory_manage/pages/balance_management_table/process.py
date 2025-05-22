import pandas as pd
from .column_types import apply_column_types
from utils.config_loader import get_template_config
from logic.manage.utils.load_template import load_master_and_template
import pandas as pd
from utils.config_loader import get_template_config


def processor_func(dfs: dict) -> pd.DataFrame:
    """
    データ処理を行う関数

    Args:
        dfs (dict): 各種データフレームを含む辞書
            - shipping: 出荷データ
            - yard: ヤードデータ
            - receive: 受入データ

    Returns:
        pd.DataFrame: 処理結果のデータフレーム
    """
    # 各データフレームを取得
    shipping_df = dfs.get("shipping", pd.DataFrame())

    # データ型を適用
    shipping_df = apply_column_types(shipping_df)

    # ここに必要な処理を追加
    # マスターCSVの読込
    master_csv = process1(shipping_df)

    # df_shippingからデータを取得
    df_after = process2(master_csv, shipping_df)
    return shipping_df


import numpy as np


def process1(df: pd.DataFrame) -> pd.DataFrame:

    # マスターCSVの読込
    config = get_template_config()["balance_management_table"]
    master_path = config["master_csv_path"]["balance_management_table"]
    master_csv = load_master_and_template(master_path)

    # データ型をintにする
    # for col in master_csv.columns:
    #     master_csv[col] = master_csv[col].astype(str).where(master_csv[col].notna(), np.nan)

    return master_csv


import pandas as pd
from typing import List


def process2(master_df: pd.DataFrame, shipping_df: pd.DataFrame) -> pd.DataFrame:
    """
    shipping_dfから master_df の条件に基づいて集計を行う。

    Args:
        master_df (pd.DataFrame): 集計条件を含むマスターデータ
        shipping_df (pd.DataFrame): 出荷データ

    Returns:
        pd.DataFrame: [大項目, 中項目, 件数, 数量, 金額]の集計結果
    """
    # 集計条件として使用するカラム（大項目・中項目は除外）
    condition_cols: List[str] = [
        col for col in master_df.columns if col not in ["大項目", "中項目"]
    ]
    results = []

    for _, condition_row in master_df.iterrows():
        # 条件適用：NaN以外のカラムだけを使う
        mask = pd.Series(True, index=shipping_df.index)
        for col in condition_cols:
            value = condition_row[col]
            if pd.notna(value) and col in shipping_df.columns:
                mask &= shipping_df[col].astype(str) == str(int(value))

        # 該当データで集計
        matched_df = shipping_df[mask]
        if not matched_df.empty:
            results.append(
                {
                    "大項目": condition_row["大項目"],
                    "中項目": condition_row["中項目"],
                    "件数": len(matched_df),
                    "数量": (
                        matched_df["数量"].sum() if "数量" in matched_df.columns else 0
                    ),
                    "金額": (
                        matched_df["金額"].sum() if "金額" in matched_df.columns else 0
                    ),
                }
            )

    # 結果まとめ
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(["大項目", "中項目"])

    return result_df
