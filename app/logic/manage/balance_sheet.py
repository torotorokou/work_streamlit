import pandas as pd

# from utils.config_loader import load_config_json
from utils.logger import app_logger
from logic.manage.processors.balance_sheet.balance_sheet_fact import (
    process_factory_report,
)
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template

# from logic.manage.utils.csv_loader import load_filtered_dataframe
from utils.config_loader import get_template_config


# 処理の統合
def process(dfs: dict) -> pd.DataFrame:
    logger = app_logger()
    """
    Streamlitの選択に基づき、工場日報（処分パート）を処理するエントリーポイント関数。
    """

    logger = app_logger()

    # --- テンプレート設定の取得 ---
    template_key = "balance_sheet"
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    csv_keys = template_config["required_files"]
    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    # --- CSVの読み込み ---
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_receive = df_dict.get("receive")
    df_shipping = df_dict.get("shipping")
    df_yard = df_dict.get("yard")

    # --- 個別処理 ---
    logger.info("▶️ 搬出量データ処理開始")
    master_csv_shobun = process_shobun(df_shipping)

    # # --- 個別処理 ---
    # logger.info("▶️ 工場日報からの読込")
    # master_csv_shobun = process_factory_report(dfs)

    # logger.info("▶️ 出荷有価データ処理開始")
    # master_csv_yuka = process_yuuka(df_yard, df_shipping)

    # logger.info("▶️ 出荷ヤードデータ処理開始")
    # master_csv_yard = process_yard(df_yard, df_shipping)

    # # --- 結合 ---
    # logger.info("🧩 各処理結果を結合中...")
    # combined_df = pd.concat(
    #     [master_csv_yuka, master_csv_shobun, master_csv_yard], ignore_index=True
    # )

    # # --- 合計・総合計行の追加/更新 ---
    # combined_df = generate_summary_dataframe(combined_df)

    # # 日付の挿入
    # combined_df = date_format(combined_df, df_shipping)

    # # --- セル行順にソート ---
    # combined_df = sort_by_cell_row(combined_df, cell_col="セル")

    # logger.debug("\n" + combined_df.to_string())

    # # --- インデックスをリセットして返す ---
    # return combined_df.reset_index(drop=True)


# def process_balance_sheet():
