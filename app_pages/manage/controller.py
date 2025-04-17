import streamlit as st
from datetime import datetime

from logic.detect_csv import detect_csv_type
from logic.manage.utils.upload_handler import handle_uploaded_files
from components.ui_message import show_warning_bubble
from app_pages.manage.view import render_file_upload_section
from app_pages.manage.view import render_status_message_ui
from logic.controllers.csv_controller import prepare_csv_data
from logic.manage.utils.processor import process_template_to_excel
from components.custom_button import centered_button
from app_pages.manage.view import render_manage_page
from utils.config_loader import (
    get_csv_date_columns,
    get_csv_label_map,
    get_required_files_map,
    get_template_descriptions,
    get_template_dict,
    get_path_config,
)


def manage_work_controller():
    # --- 設定を取得 ---
    template_dict = get_template_dict()
    template_descriptions = get_template_descriptions()
    required_files = get_required_files_map()
    csv_label_map = get_csv_label_map()
    date_columns = get_csv_date_columns()
    header_csv_path = get_path_config()["csv"]["receive_header_definition"]

    # --- サイドバーからテンプレート選択 ---
    selected_template_label = render_manage_page(
        template_dict,
        template_descriptions,
    )

    # --- 選択されたテンプレートに応じて必要ファイルキーを取得 ---
    selected_template = template_dict.get(selected_template_label)
    required_keys = required_files.get(selected_template, [])

    # --- ファイルアップロードUI表示 & 取得 ---
    uploaded_files = render_file_upload_section(required_keys, csv_label_map)

    # --- CSVファイルの妥当性確認（毎回確認）---
    validated_files = handle_uploaded_files(
        required_keys, csv_label_map, header_csv_path
    )

    # --- アップロードされていないファイルを確認 ---
    missing_keys = [k for k in required_keys if validated_files.get(k) is None]
    all_uploaded = len(missing_keys) == 0

    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")

    # ✅ 毎回更新されたアップロード状態に応じてボタンを切り替える
    if centered_button("📊 書類作成", disabled=not all_uploaded):
        st.markdown("---")
        progress = st.progress(0)
        progress.progress(10, "📥 ファイルを処理中...")

        dfs = prepare_csv_data(uploaded_files, date_columns, selected_template)
        config = get_path_config()
        output_excel = process_template_to_excel(
            selected_template, dfs, csv_label_map, config
        )

        today_str = datetime.now().strftime("%Y%m%d")
        file_name = f"{selected_template}_{today_str}.xlsx"

        render_status_message_ui(
            file_ready=True, file_name=file_name, output_excel=output_excel
        )

    else:
        # アップロード状況の表示
        uploaded_count = len(required_keys) - len(missing_keys)
        total_count = len(required_keys)

        render_status_message_ui(
            file_ready=False, uploaded_count=uploaded_count, total_count=total_count
        )
