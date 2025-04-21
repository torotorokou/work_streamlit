import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template


# 処理の統合
def process(dfs: dict, csv_label_map: dict) -> pd.DataFrame:

    logger = app_logger()
    # 対象CSVの読み込み
    csv_name = ["shipping", "yard"]
    logger.info("Processの処理に入る")
    df_dict = load_all_filtered_dataframes(dfs, csv_name)

    # 集計処理ステップ（明示的）
    df_shipping = df_dict.get(csv_name[0])
    df_yard = df_dict.get(csv_name[1])

    # マスターファイルとテンプレートの読み込み
    master_path = get_template_config()["factory_report"]["master_csv_path"]["shobun"]
    master_csv = load_master_and_template(master_path)

    # 集計処理ステップ
    master_csv_shipping = process_shipping(df_shipping, master_csv_shipping)
    master_csv_yuka = process_yuka(df_yard, master_csv_yuka)
    master_csv_yard = process_yard(df_yard, master_csv_yard)

    return master_csv


# CSVごとにプロセスを分ける