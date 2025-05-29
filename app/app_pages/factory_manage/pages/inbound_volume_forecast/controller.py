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


def factory_manage_controller():
    selected_template = "monitor"
    # --- 必要ファイルキーを取得 ---
    required_keys = load_factory_required_files()[selected_template]

    # 🔽 再計算用
    if "selected_template_cache" not in st.session_state:
        st.session_state.selected_template_cache = selected_template
    elif st.session_state.selected_template_cache != selected_template:
        st.session_state.process_step = None
        st.session_state.df_result = None
        st.session_state.extracted_date = None
        st.session_state.selected_template_cache = selected_template

    # --- ファイルアップロードUI表示 & 取得 ---
    render_shipping_upload_section()

    # --- 整合性チェック（session_stateから取得 → validate）
    uploaded_files = handle_uploaded_files(required_keys)

    # --- アップロード状態チェック（単一ファイル）
    uploaded_file = uploaded_files.get("shipping")
    all_uploaded, missing_key = check_single_file_uploaded(uploaded_file, "shipping")
    print(all_uploaded, missing_key)

    # --- セッション初期化（ファイルが足りないとき）
    if not all_uploaded and "process_step" in st.session_state:
        st.session_state.process_step = None
        st.session_state.dfs = None
        st.session_state.df_result = None
        st.session_state.extracted_date = None

    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.markdown("---")
