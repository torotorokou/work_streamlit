import streamlit as st
from logic.detect_csv import detect_csv_type
from logic.manage.upload_handler import handle_uploaded_files
from components.ui_message import show_warning_bubble
from app_pages.manage.view import render_status_message
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
    from app_pages.manage.view import render_file_upload_section

    uploaded_files = render_file_upload_section(required_keys, csv_label_map)

    # --- ヘッダーチェック ---
    validated_files = handle_uploaded_files(
        required_keys, csv_label_map, header_csv_path
    )


    # --- アップロード状態チェック ---
    missing_keys = [k for k in required_keys if uploaded_files.get(k) is None]

    # --- 帳票作成またはステータス表示に分岐 ---
    render_status_message(
        missing_keys,
        required_keys,
        uploaded_files,
        date_columns,
        selected_template,
        csv_label_map,
        get_path_config()  # 必要ならconfig渡す
    )
