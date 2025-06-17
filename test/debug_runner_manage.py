import os

os.chdir("/work/app")

import pandas as pd
from utils.logger import app_logger
from utils.config_loader import get_template_config
from logic.manage.utils.csv_loader import load_all_filtered_dataframes
from logic.controllers.csv_controller import apply_expected_dtypes
from logic.manage.utils.load_template import load_master_and_template
from logic.manage.factory_report import process as process_factory
from logic.manage.balance_sheet import process as process_balance_sheet
from logic.manage.management_sheet import process as process_manage_sheet
from logic.manage.block_unit_price import process as process_block_unit_price
import streamlit as st

# Streamlitのセッション状態を初期化
if "process_mini_step" not in st.session_state:
    st.session_state.process_mini_step = 0
if "block_unit_price_confirmed" not in st.session_state:
    st.session_state.block_unit_price_confirmed = False
if "block_unit_price_transport_map" not in st.session_state:
    st.session_state.block_unit_price_transport_map = {}


# 処理の統合
def run_debug_process(template_key) -> pd.DataFrame:
    logger = app_logger()

    # デバッグ用のファイルパス（存在するもののみ定義）
    debug_file_paths = {
        "receive": "/work/app/data/input/debug_receive.parquet",
        "shipping": "/work/app/data/input/debug_shipping.parquet",
        "yard": "/work/app/data/input/debug_yard.parquet",
    }

    # --- テンプレート定義読み込み ---
    template_config = get_template_config()[template_key]
    template_name = template_config["key"]
    required_keys = template_config.get("required_files", [])
    optional_keys = template_config.get("optional_files", [])
    csv_keys = required_keys + optional_keys

    logger.info(f"[テンプレート設定読込] key={template_key}, files={csv_keys}")

    # --- dfs を柔軟に構築（optionalは存在チェック）---
    dfs = {}
    for key in csv_keys:
        path = debug_file_paths.get(key)
        if path:
            try:
                dfs[key] = pd.read_parquet(path)
            except FileNotFoundError:
                logger.warning(
                    f"[DEBUG] optionalファイル {key} が見つかりません: {path}"
                )
                dfs[key] = None
        else:
            dfs[key] = None

    # --- 型変換・フィルタリング処理 ---
    dfs = apply_expected_dtypes(dfs, template_key)
    df_dict = load_all_filtered_dataframes(dfs, csv_keys, template_name)

    return df_dict


def debug():
    # --- テンプレート設定の取得 ---
    template_key = "balance_sheet"
    logger = app_logger()
    dfs_after = run_debug_process(template_key)

    logger.info("デバッグ作業開始")

    # 各ステップを手動で進める

    process = {
        "factory_report": process_factory,
        "balance_sheet": process_balance_sheet,
        "manage_sheet": process_manage_sheet,
        "block_unit_price": process_block_unit_price,
    }

    func = process.get(template_key)
    if func:
        result = func(dfs_after)
    else:
        raise ValueError(f"無効なテンプレートキー: {template_key}")

    if result is None and st.session_state.process_mini_step == 1:
        # Step 1の場合、運搬業者の選択を自動化
        st.session_state.block_unit_price_confirmed = True
        result = process.get(template_key)

    if result is None and st.session_state.process_mini_step == 2:
        # Step 1の場合、運搬業者の選択を自動化
        st.session_state.block_unit_price_confirmed = True
        result = process.get(template_key)

    print(f"Current mini_step: {st.session_state.process_mini_step}")
    print(f"Result: {result is not None}")
    return


debug()
