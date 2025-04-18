# ✅ 標準ライブラリ
import time

# ✅ サードパーティ
import streamlit as st

# ✅ プロジェクト内 - components（UI共通パーツ）
from components.custom_button import centered_button, centered_download_button
# from components.ui_message import show_warning_bubble

# ✅ プロジェクト内 - view（UIビュー）
from app_pages.manage.view import (
    render_file_upload_section,
    render_manage_page,
    # render_status_message_ui,
)

# ✅ プロジェクト内 - logic（処理・データ変換など）
from logic.manage import template_processors
from logic.controllers.csv_controller import prepare_csv_data
# from logic.detect_csv import detect_csv_type
from logic.manage.utils.upload_handler import handle_uploaded_files

# ✅ プロジェクト内 - utils（共通ユーティリティ）
from utils.logger import app_logger
from utils.write_excel import write_values_to_template
from utils.config_loader import (
    get_csv_date_columns,
    get_csv_label_map,
    get_required_files_map,
    get_template_descriptions,
    get_template_dict,
    get_path_config,
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

    # 書類作成
    # --- ステータス表示 ---
    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")

        # if not missing_keys:
        #     st.success("✅ 必要なファイルがすべてアップロードされました！")

        st.markdown("---")
        if centered_button("📊 書類作成"):
            progress = st.progress(0)

            progress.progress(10, "📥 ファイルを処理中...")
            time.sleep(0.3)

            # dfsとcsv日付の作成
            dfs, extracted_date = prepare_csv_data(uploaded_files, date_columns, selected_template)
            extracted_date = extracted_date[0].strftime("%Y%m%d")
            logger.info("dfsの読込完了")

            processor_func = template_processors.get(selected_template)

            # テンプレートに従い、処理実行
            if processor_func:
                progress.progress(40, "🧮 データを計算中...")
                time.sleep(0.3)

                # 個々のprocessにより、dfを取得
                df = processor_func(dfs, csv_label_map)

                progress.progress(70, "📄 テンプレートに書き込み中...")
                time.sleep(0.3)

                # テンプレートへの書き込み
                template_path = get_template_config()[selected_template]["template_excel_path"]
                output_excel = write_values_to_template(df, template_path,extracted_date)

                progress.progress(90, "✅ 整理完了")
                time.sleep(0.3)
                progress.progress(100)

                # ダウンロードボタン表示
                st.info("✅ ファイルが生成されました。下のボタンからダウンロードできます👇")
                centered_download_button(
                    label="📥 Excelファイルをダウンロード",
                    data=output_excel.getvalue(),
                    file_name=f"{selected_template_label}_{extracted_date}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    else:

        uploaded_count = len(required_keys) - len(missing_keys)
        total_count = len(required_keys)

        st.progress(uploaded_count / total_count)
        st.info(f"📥 {uploaded_count} / {total_count} ファイルがアップロードされました")
