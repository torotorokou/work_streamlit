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
from logic.manage.factory_report import process as process_factory
from logic.manage.utils.excel_tools import create_label_rows_generic, sort_by_cell_row
from utils.value_setter import set_value_fast, set_value_fast_safe
from logic.manage.utils.summary_tools import summary_apply
from logic.manage.balance_sheet import process as process_balance_sheet
from logic.manage.management_sheet import process as process_manage_sheet
from logic.manage.block_unit_price import process as process_block_unit_price

# factory_manageのコントローラーをインポート
try:
    from app_pages.factory_manage.controller import factory_manage_work_controller
except ImportError as e:
    print(f"factory_manage_work_controllerのインポートに失敗: {e}")
    factory_manage_work_controller = None

def run_debug_process() -> pd.DataFrame:
    logger = app_logger()

    debug_receive = "/work/app/data/input/debug_receive.parquet"
    debug_shipping = "/work/app/data/input/debug_shipping.parquet"
    debug_yard = "/work/app/data/input/debug_yard.parquet"

    dfs = {
        "receive": pd.read_parquet(debug_receive),
        "shipping": pd.read_parquet(debug_shipping),
        "yard": pd.read_parquet(debug_yard),
    }

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
        "receive": df_receive,
        "shipping": df_shipping,
        "yard": df_yard,
    }
    return dfs_after

if __name__ == "__main__":
    logger = app_logger()
    logger.info("デバッグ作業開始")
    dfs_after = run_debug_process()
    process_block_unit_price(dfs_after)

    # factory_manageの動作確認
    if factory_manage_work_controller is not None:
        try:
            print("factory_manage_work_controllerを呼び出します")
            # 必要に応じて引数を渡してください
            factory_manage_work_controller()
            print("factory_manage_work_controllerの呼び出し成功")
        except Exception as e:
            print(f"factory_manage_work_controllerの実行中にエラー: {e}")
    else:
        print("factory_manage_work_controllerがインポートできませんでした")