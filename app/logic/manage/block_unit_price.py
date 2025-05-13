from utils.logger import app_logger
import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.manage.utils.load_template import load_master_and_template


def process(dfs):
    logger = app_logger()


    # --- テンプレート設定の取得 ---
    template_key = "block_unit_price"
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    csv_keys = template_config["required_files"]
    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    # --- CSVの読み込み ---
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_shipping = df_dict.get("shipping")


    # --- 個別処理 ---
    logger.info("▶️ プロセス1")
    master_csv = process1(df_shipping)



    return master_csv


def process1(df_shipping):

    # --- ① マスターCSVの読み込み ---
    config = get_template_config()["block_unit_price"]
    master_path = config["master_csv_path"]["vendor_code"]
    master_csv = load_master_and_template(master_path)


    # 必要行の抜き出し
    df_after = df_shipping[df_shipping["業者CD"].isin(master_csv["業者CD"])]




    return