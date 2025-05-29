import streamlit as st

# ✅ 標準ライブラリ

# ✅ サードパーティ
import pandas as pd

# ✅ プロジェクト内 - components（UI共通パーツ）
from components.custom_button import centered_button
from components.custom_progress_bar import CustomProgressBar

# ✅ プロジェクト内 - view（UIビュー）

# ✅ プロジェクト内 - logic（処理・データ変換など）
from logic.manage.utils.upload_handler import handle_uploaded_files

# ✅ プロジェクト内 - utils（共通ユーティリティ）
from utils.debug_tools import save_debug_parquets
from utils.config_loader import (
    get_csv_label_map,
)

from utils.config_loader import load_factory_required_files
from app_pages.factory_manage.pages.balance_management_table.process import (
    processor_func,
)
from components.custom_button import centered_download_button
from io import BytesIO
from utils.check_uploaded_csv import (
    render_csv_upload_section,
    check_single_file_uploaded,
)
from logic.factory_manage.modelver2_1day.make_df import make_sql_db


def csv_controller():
    selected_template = "inbound_volume"
    # --- 必要ファイルキーを取得 ---
    required_keys = load_factory_required_files()[selected_template]

    # --- ファイルアップロードUI表示 & 取得 ---
    csv_file_type = "receive"
    render_csv_upload_section(csv_file_type)

    # --- 整合性チェック（session_stateから取得 → validate）
    uploaded_files = handle_uploaded_files(required_keys)

    # --- アップロード状態チェック（単一ファイル）
    uploaded_file = uploaded_files.get(csv_file_type)
    all_uploaded, missing_key = check_single_file_uploaded(uploaded_file, csv_file_type)
    print(all_uploaded, missing_key)

    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.markdown("---")

        if centered_button("⏩ CSVをアップロード"):
            # --- CSV読み込み ---
            df = pd.read_csv(uploaded_file)
            make_sql_db(df)
            st.rerun()
