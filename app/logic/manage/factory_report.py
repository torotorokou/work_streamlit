import pandas as pd
from utils.logger import app_logger, debug_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.processors.factory_report.factory_report_shobun import process_shobun
from logic.manage.processors.factory_report.factory_report_yuuka import process_yuuka
from logic.manage.processors.factory_report.factory_report_yard import process_yard
from logic.manage.processors.factory_report.make_cell_num import make_cell_num
from logic.manage.utils.excel_tools import sort_by_cell_row
from logic.manage.processors.factory_report.etc import (
    generate_summary_dataframe,
    upsert_summary_row,
    date_format,
)

# from logic.manage.utils.load_template import load_master_and_template
# from utils.date_tools import to_japanese_era, to_japanese_month_day
# from utils.value_setter import set_value_fast, set_value_fast_safe


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

    # セル番号の設定
    combined_df= make_cell_num(combined_df)


    # --- 合計・総合計行の追加/更新 ---
    combined_df = generate_summary_dataframe(combined_df)

    # 日付の挿入
    combined_df = date_format(combined_df, df_shipping)

    # --- セル行順にソート ---
    combined_df = sort_by_cell_row(combined_df, cell_col="セル")

    # --- インデックスをリセットして返す ---
    return combined_df.reset_index(drop=True)
