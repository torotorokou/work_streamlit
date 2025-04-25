import pandas as pd
from utils.logger import app_logger, debug_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.processors.factory_report_shobun import process_shobun
from logic.manage.processors.factory_report_yuuka import process_yuuka
from logic.manage.processors.factory_report_yard import process_yard
from logic.manage.utils.excel_tools import sort_by_cell_row
from logic.manage.utils.load_template import load_master_and_template
from utils.date_tools import to_japanese_era, to_japanese_month_day
from utils.value_setter import set_value_fast
from logic.manage.utils.summary_tools import (
    write_sum_to_target_cell,
    summarize_value_by_cell_with_label,
)
from typing import Optional


def process(dfs: dict) -> pd.DataFrame:
    """
    Streamlitの選択に基づき、工場日報（処分パート）を処理するエントリーポイント関数。
    """

    logger = app_logger()
    deb_logger = debug_logger()

    # --- テンプレート設定の取得 ---
    template_key = "factory_report"
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    csv_keys = template_config["required_files"]
    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    # --- CSVの読み込み ---
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_shipping = df_dict.get("shipping")
    df_yard = df_dict.get("yard")

    # --- 個別処理 ---
    logger.info("▶️ 出荷処分データ処理開始")
    master_csv_shobun = process_shobun(df_shipping)

    logger.info("▶️ 出荷有価データ処理開始")
    master_csv_yuka = process_yuuka(df_yard, df_shipping)

    logger.info("▶️ 出荷ヤードデータ処理開始")
    master_csv_yard = process_yard(df_yard, df_shipping)

    # --- 結合 ---
    logger.info("🧩 各処理結果を結合中...")
    combined_df = pd.concat(
        [master_csv_yuka, master_csv_shobun, master_csv_yard], ignore_index=True
    )

    # --- 合計・総合計行の追加/更新 ---
    combined_df = generate_summary_dataframe(combined_df)

    # 日付の挿入
    combined_df = date_format(combined_df, df_shipping)

    # --- セル行順にソート ---
    combined_df = sort_by_cell_row(combined_df, cell_col="セル")

    logger.debug("\n" + combined_df.to_string())

    # --- インデックスをリセットして返す ---
    return combined_df.reset_index(drop=True)


def generate_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    config = get_template_config()["factory_report"]
    etc_path = config["master_csv_path"]["etc"]
    etc_csv = load_master_and_template(etc_path)

    # 1. コピーして元dfを保護
    df_sum = df.copy()

    # 2. 値列を数値に変換（NaN対応）
    df_sum["値"] = pd.to_numeric(df_sum["値"], errors="coerce")

    # 3. カテゴリ別の合計
    category_sum = df_sum.groupby("カテゴリ")["値"].sum()

    # 4. 総合計
    total_sum = df_sum["値"].sum()

    # 5. テンプレに合計をマージ
    def assign_sum(row):
        if "ヤード" in row["大項目"] and "処分" not in row["大項目"]:
            return category_sum.get("ヤード", 0.0)
        elif "処分" in row["大項目"] and "ヤード" not in row["大項目"]:
            return category_sum.get("処分", 0.0)
        elif "有価" in row["大項目"]:
            return category_sum.get("有価", 0.0)
        elif "総合計" in row["大項目"]:
            return total_sum
        return row["値"]

    etc_csv["値"] = etc_csv.apply(assign_sum, axis=1)

    # 6. 合計_処分ヤード = 処分 + ヤード の合算
    mask_shobun_yard = etc_csv["大項目"] == "合計_処分ヤード"
    val_shobun = etc_csv.loc[etc_csv["大項目"] == "合計_処分", "値"].values
    val_yard = etc_csv.loc[etc_csv["大項目"] == "合計_ヤード", "値"].values

    if val_shobun.size > 0 and val_yard.size > 0:
        etc_csv.loc[mask_shobun_yard, "値"] = val_shobun[0] + val_yard[0]

    # 7. 元dfとetcの結合（縦方向）
    df_combined = pd.concat([df, etc_csv], ignore_index=True)

    return df_combined


def upsert_summary_row(
    df: pd.DataFrame,
    label: str,
    value: float,
    value_col: str = "値",
    label_col: str = "大項目",
) -> pd.DataFrame:
    """
    指定ラベルの行が存在すれば値を更新し、存在しなければセル列は空のまま新規行として追加する。
    ※ セル列はすでに記入済みである前提

    Parameters:
        df (pd.DataFrame): 対象データフレーム
        label (str): "大項目" のラベル名（例："総合計" など）
        value (float): 書き込む値
        value_col (str): 値の列名
        label_col (str): ラベルの列名

    Returns:
        pd.DataFrame: 更新済みのDataFrame
    """
    mask = df[label_col] == label

    if mask.any():
        # 既存行があるなら値だけ更新
        df.loc[mask, value_col] = value
    else:
        # セル列には何も書かず追加（補完は別途）
        new_row = {
            label_col: label,
            value_col: value,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df


def date_format(master_csv, df_shipping):
    today = pd.to_datetime(df_shipping["伝票日付"].dropna().iloc[0])

    master_columns_keys = ["大項目"]
    key_name = ["和暦"]
    set_value_fast(master_csv, master_columns_keys, key_name, to_japanese_era(today))

    master_columns_keys = ["大項目"]
    key_name = ["月日"]
    set_value_fast(
        master_csv, master_columns_keys, key_name, to_japanese_month_day(today)
    )

    return master_csv
