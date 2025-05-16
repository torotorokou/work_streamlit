# ✅ 標準ライブラリ
import time

# ✅ サードパーティ
import streamlit as st

# ✅ プロジェクト内 - components（UI共通パーツ）
from components.custom_button import centered_button, centered_download_button

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


def manage_work_controller():
    logger = app_logger()

    # --- 設定を取得 ---
    template_dict = get_template_dict()
    template_descriptions = get_template_descriptions()
    required_files = get_required_files_map()
    csv_label_map = get_csv_label_map()
    date_columns = get_csv_date_columns()

    # --- UI:テンプレート選択 ---
    selected_template_label = render_manage_page(
        template_dict,
        template_descriptions,
    )
    selected_template = template_dict.get(selected_template_label)

    # --- 必要ファイルキーを取得 ---
    required_keys = required_files.get(selected_template, [])

    # --- ファイルアップロードUI表示 & 取得 ---
    uploaded_files = render_file_upload_section(required_keys, csv_label_map)

    # --- CSVファイルの妥当性確認（毎回確認）---
    validated_files = handle_uploaded_files(required_keys, csv_label_map)

    # --- アップロードされていないファイルを確認 ---
    all_uploaded, missing_keys = check_missing_files(uploaded_files, required_keys)

    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.markdown("---")

        # --- ステップ制御 ---
        step = st.session_state.get("process_step", 0)

        if step == 0:
            if centered_button("⏩ 書類作成を開始する"):
                dfs, extracted_date = prepare_csv_data(
                    uploaded_files, date_columns, selected_template
                )
                st.session_state.dfs = dfs
                st.session_state.extracted_date = extracted_date[0].strftime("%Y%m%d")
                st.session_state.process_step = 1
                st.rerun()

        elif step == 1:
            processor_func = template_processors.get(selected_template)
            df_result = processor_func(st.session_state.dfs)

            if df_result is None:
                st.stop()  # UI選択画面などで中断されている

            st.session_state.df_result = df_result
            st.session_state.process_step = 2
            st.rerun()

        elif step == 2:
            st.success("✅ 書類作成完了")
            df_result = st.session_state.df_result
            template_path = get_template_config()[selected_template]["template_excel_path"]
            output_excel = write_values_to_template(
                df_result, template_path, st.session_state.extracted_date
            )
            centered_download_button(
                label="📥 Excelファイルをダウンロード",
                data=output_excel.getvalue(),
                file_name=f"{selected_template_label}_{st.session_state.extracted_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    else:
        uploaded_count = len(required_keys) - len(missing_keys)
        total_count = len(required_keys)

        st.progress(uploaded_count / total_count)
        st.info(f"📥 {uploaded_count} / {total_count} ファイルがアップロードされました")