import os

os.chdir("/work/app")

import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.controllers.csv_controller import apply_expected_dtypes
from logic.manage.utils.load_template import load_master_and_template
from IPython.display import display
import re
from logic.manage.factory_report import process
from logic.manage.utils.excel_tools import create_label_rows_generic, sort_by_cell_row
from utils.value_setter import set_value_fast, set_value_fast_safe
from logic.manage.utils.summary_tools import summary_apply
from logic.manage.factory_report import process as process_factory
from logic.manage.balance_sheet import process as process_balance_sheet
from logic.manage.management_sheet import process as process_manage_sheet
from logic.manage.block_unit_price import process as process_block_unit_price

# 処理の統合
def run_debug_process() -> pd.DataFrame:
    logger = app_logger()

    # 表示ラベルマップ（処理対象名として使う）
    # csv_label_map = {"yard": "ヤード一覧", "shipping": "出荷一覧", "receive": "受入一覧"}

    debug_receive = "/work/app/data/input/debug_receive.parquet"
    debug_shipping = "/work/app/data/input/debug_shipping.parquet"
    debug_yard = "/work/app/data/input/debug_yard.parquet"

    dfs = {
        "receive": pd.read_parquet(debug_receive),
        "shipping": pd.read_parquet(debug_shipping),
        "yard": pd.read_parquet(debug_yard),
    }

    # dfs
    df_shipping = dfs["shipping"]
    df_yard = dfs["yard"]
    df_receive = dfs["receive"]


    # --- テンプレート設定の取得 ---
    template_key = "block_unit_price"

    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    csv_keys = template_config["required_files"]
    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    # --- CSVの調整・読み込み ---
    dfs = apply_expected_dtypes(dfs, template_key)
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)
    df_receive = df_dict.get("receive")
    df_shipping = df_dict.get("shipping")
    df_yard = df_dict.get("yard")

    dfs_after = {
        "receive":df_receive,
        "shipping": df_shipping,
        "yard":df_yard,
    }
    return dfs_after


logger = app_logger()
dfs_after = run_debug_process()

logger.info("デバッグ作業開始")
process_block_unit_price(dfs_after)
