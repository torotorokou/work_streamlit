import streamlit as st

# ✅ 標準ライブラリ
import time

# ✅ サードパーティ
import streamlit as st

# ✅ プロジェクト内 - components（UI共通パーツ）
from components.custom_button import centered_button, centered_download_button
from components.custom_progress_bar import CustomProgressBar

# ✅ プロジェクト内 - view（UIビュー）
from app_pages.manage.view import (
    render_file_upload_section,
    render_manage_page,
)

# ✅ プロジェクト内 - logic（処理・データ変換など）
from logic.manage import template_processors
from logic.controllers.csv_controller import prepare_csv_data
from logic.manage.utils.upload_handler import handle_uploaded_files
from logic.manage.utils.file_validator import check_missing_files

# ✅ プロジェクト内 - utils（共通ユーティリティ）
from utils.progress_helper import update_progress
from utils.logger import app_logger
from utils.write_excel import write_values_to_template
from utils.debug_tools import save_debug_parquets
from utils.config_loader import (
    get_csv_date_columns,
    get_csv_label_map,
    get_required_files_map,
    get_template_descriptions,
    get_template_dict,
    get_template_config,
)

from utils.config_loader import load_factory_required_files
from utils.config_loader import get_csv_label_map


def render_waste_management_table():
    st.subheader("🗑 工場収支モニタリング表")
    st.write("処理実績や分類別の集計を表示します。")

    # --- 必要ファイルキーを取得 ---
    required_keys = load_factory_required_files()["monitor"]
    csv_label_map = get_csv_label_map()

    # --- ファイルアップロードUI表示 & 取得 ---
    st.markdown("### 📂 CSVファイルのアップロード")
    st.info("以下のファイルをアップロードしてください。")
    uploaded_files = render_file_upload_section(required_keys, csv_label_map)

    # --- CSVファイルの妥当性確認（毎回確認）---
    handle_uploaded_files(required_keys, csv_label_map)

    # --- アップロードされていないファイルを確認 ---
    all_uploaded, missing_keys = check_missing_files(uploaded_files, required_keys)

    # --- アップロードされていないファイルを確認 ---
    all_uploaded, missing_keys = check_missing_files(uploaded_files, required_keys)

    # ✅ ファイルがなくなった場合はセッション状態をリセット
    if not all_uploaded and "process_step" in st.session_state:
        st.session_state.process_step = None
        st.session_state.dfs = None
        st.session_state.df_result = None
        st.session_state.extracted_date = None

    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.markdown("---")
