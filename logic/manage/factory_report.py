import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.processors.factory_report_shobun import process_shobun
from logic.manage.processors.factory_report_yuuka import process_yuuka
from logic.manage.processors.factory_report_yard import process_yard
from logic.manage.utils.excel_tools import sort_by_cell_row
from typing import Optional


def process(dfs: dict) -> pd.DataFrame:
    """
    Streamlitの選択に基づき、工場日報（処分パート）を処理するエントリーポイント関数。
    """

    logger = app_logger()

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
        [master_csv_yuka, master_csv_shobun, master_csv_yard],
        ignore_index=True
    )

    # --- セル行順にソート ---
    combined_df = sort_by_cell_row(combined_df, cell_col="セル")

    # --- 合計・総合計行の追加/更新 ---
    combined_df = sum_array(combined_df)

    # --- インデックスをリセットして返す ---
    return combined_df.reset_index(drop=True)


def sum_array(df: pd.DataFrame) -> pd.DataFrame:
    disposal = df.loc[df["大項目"] == "合計_処分", "値"].sum()
    yard = df.loc[df["大項目"] == "合計_ヤード", "値"].sum()
    value_disposal_yard = disposal + yard

    df = upsert_summary_row(df, "合計_処分ヤード", value_disposal_yard)

    valuable = df.loc[df["大項目"] == "合計_有価", "値"].sum()
    total = value_disposal_yard + valuable

    df = upsert_summary_row(df, "総合計", total)

    return df


def upsert_summary_row(
    df: pd.DataFrame,
    label: str,
    value: float,
    value_col: str = "値",
    label_col: str = "大項目"
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
