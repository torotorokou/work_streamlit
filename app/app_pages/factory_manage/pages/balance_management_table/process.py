import pandas as pd
import numpy as np
from utils.cleaners import enforce_dtypes, strip_whitespace
from utils.config_loader import (
    get_template_config,
    get_expected_dtypes_by_template,
    get_required_columns_definition,
)
from logic.manage.utils.load_template import load_master_and_template


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
    shipping_df = make_csv(dfs)

    # マスターCSVの読込
    master_csv = process1(shipping_df)

    # df_shippingからデータを取得
    df_after = process2(master_csv, shipping_df)
    return shipping_df


def make_csv(dfs):
    # 各データフレームを取得
    shipping_df = dfs.get("shipping", pd.DataFrame())

    # 必要なカラムのリストを取得
    cul_list = get_required_shipping_columns()

    # 必要なカラムが存在するか確認
    missing_columns = [col for col in cul_list if col not in shipping_df.columns]
    if missing_columns:
        raise ValueError(
            f"以下の必要なカラムが出荷データに存在しません: {missing_columns}"
        )

    # 必要なカラムのみに絞る
    shipping_df = shipping_df[cul_list]

    # 空白除去
    shipping_df = strip_whitespace(shipping_df)

    # テンプレートに基づく型定義を取得
    expected_dtypes = get_expected_dtypes_by_template("balance_management_table")
    shipping_dtypes = expected_dtypes.get("shipping", {})

    # データ型を適用
    if shipping_dtypes:
        shipping_df = enforce_dtypes(shipping_df, shipping_dtypes)

    return shipping_df


def process1(df: pd.DataFrame) -> pd.DataFrame:
    # マスターCSVの読込
    config = get_template_config()["balance_management_table"]
    master_path = config["master_csv_path"]["balance_management_table"]
    master_csv = load_master_and_template(master_path)

    # テンプレートに基づく型定義を取得
    expected_dtypes = get_expected_dtypes_by_template("balance_management_table")
    master_dtypes = expected_dtypes.get("master", {})

    # 空白除去
    master_csv = strip_whitespace(master_csv)

    # データ型を適用
    if master_dtypes:
        master_csv = enforce_dtypes(master_csv, master_dtypes)
    else:
        # マスター定義がない場合のデフォルト処理
        for col in master_csv.columns:
            if col in ["大項目", "中項目"]:
                master_csv[col] = (
                    master_csv[col].astype(str).where(master_csv[col].notna(), "")
                )
            else:
                master_csv[col] = (
                    master_csv[col].astype(str).where(master_csv[col].notna(), np.nan)
                )

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
    # デバッグ情報：入力データの確認
    print("\n=== データフレーム情報 ===")
    print("マスターデータ:")
    print(f"行数: {len(master_df)}")
    print("カラムと型:")
    print(master_df.dtypes)
    print("\n出荷データ:")
    print(f"行数: {len(shipping_df)}")
    print("カラムと型:")
    print(shipping_df.dtypes)

    # 集計条件として使用するカラム（大項目・中項目は除外）
    condition_cols: List[str] = [
        col for col in master_df.columns if col not in ["大項目", "中項目"]
    ]
    print("\n使用する条件カラム:", condition_cols)

    results = []

    for idx, condition_row in master_df.iterrows():
        print(f"\n=== マスター行 {idx} の処理 ===")
        # 条件適用：NaN以外のカラムだけを使う
        mask = pd.Series(True, index=shipping_df.index)

        for col in condition_cols:
            value = condition_row[col]
            # 値が0またはnanの場合はその条件をスキップ
            if pd.isna(value) or (isinstance(value, (int, float)) and value == 0):
                print(f"列 {col} の条件をスキップ: 値が0またはnan")
                continue

            if col in shipping_df.columns:
                print(f"\n列 {col} の比較:")
                print(f"  マスター値: {value} (型: {type(value)})")
                print(f"  出荷データの一意な値: {shipping_df[col].unique()}")

                # 比較前のマスクの真の数
                before_count = mask.sum()
                mask &= shipping_df[col] == value
                after_count = mask.sum()

                print(f"  比較前のマッチ数: {before_count}")
                print(f"  比較後のマッチ数: {after_count}")
                print(f"  この条件で除外された行数: {before_count - after_count}")

        # 該当データで集計
        matched_df = shipping_df[mask]
        print(f"\nマッチした行数: {len(matched_df)}")

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
        else:
            print("警告: 条件に一致するデータがありません")
            print("条件値:")
            for col in condition_cols:
                if pd.notna(condition_row[col]) and not (
                    isinstance(condition_row[col], (int, float))
                    and condition_row[col] == 0
                ):
                    print(
                        f"  {col}: {condition_row[col]} (型: {type(condition_row[col])})"
                    )

    # 結果まとめ
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(["大項目", "中項目"])
    else:
        print("\n警告: 最終的な結果が空です")

    return result_df


def get_required_shipping_columns() -> list:
    """
    balance_management_tableに必要な出荷データのカラムを取得する

    Returns:
        list: 必要なカラムのリスト
    """
    # required_columns_definition.yamlから必要なカラムを取得
    required_cols = get_required_columns_definition("balance_management_table")

    # shippingのカラムを取得（存在しない場合は空のリストを返す）
    shipping_cols = required_cols.get("shipping", [])

    return shipping_cols
