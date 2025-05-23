# ✅ 標準ライブラリ

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
from utils.logger import app_logger
from utils.write_excel import write_values_to_template
from utils.config_loader import (
    get_csv_date_columns,
    get_required_files_map,
    get_template_descriptions,
    get_template_dict,
    get_template_config,
)


def manage_work_controller():
    logger = app_logger()

    # --- UI:テンプレート選択 ---
    template_dict = dict(list(get_template_dict().items())[:5])

    template_descriptions = get_template_descriptions()
    selected_template_label = render_manage_page(
        template_dict,
        template_descriptions,
    )
    selected_template = template_dict.get(selected_template_label)

    # --- 必要ファイルキーを取得 ---
    required_files = get_required_files_map()
    required_keys = required_files.get(selected_template, [])

    # 🔽 再計算用
    if "selected_template_cache" not in st.session_state:
        st.session_state.selected_template_cache = selected_template
    elif st.session_state.selected_template_cache != selected_template:
        st.session_state.process_step = None
        st.session_state.df_result = None
        st.session_state.extracted_date = None
        st.session_state.selected_template_cache = selected_template

    # --- ファイルアップロードUI表示 & 取得 ---
    st.markdown("### 📂 CSVファイルのアップロード")
    st.info("以下のファイルをアップロードしてください。")
    uploaded_files = render_file_upload_section(required_keys)

    # --- CSVファイルの妥当性確認（毎回確認）---
    handle_uploaded_files(required_keys)

    # --- アップロードされていないファイルを確認 ---
    all_uploaded, missing_keys = check_missing_files(uploaded_files, required_keys)

    # ✅ ファイルがなくなった場合はセッション状態をリセット
    if not all_uploaded and "process_step" in st.session_state:
        st.session_state.process_step = None
        st.session_state.dfs = None
        st.session_state.df_result = None
        st.session_state.extracted_date = None

    if all_uploaded:
        date_columns = get_csv_date_columns()
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.markdown("---")

        # --- ステップ管理の初期化（ボタン押下前は None）---
        if "process_step" not in st.session_state:
            st.session_state.process_step = None

        # --- 書類作成ボタンを最初に表示し、押されたらステップ0へ移行 ---
        if st.session_state.process_step is None:
            if centered_button("⏩ 書類作成を開始する"):
                st.session_state.process_step = 0
                st.rerun()
            return  # ボタンが押されるまでは処理しない

        # --- ステップ制御とプログレス描画 ---
        step = st.session_state.get("process_step", 0)
        progress_bar = CustomProgressBar(
            total_steps=3, labels=["📥 読込中", "🧮 処理中", "📄 出力"]
        )

        # ✅ プログレスバーの描画
        with st.container():
            progress_bar.render(step)

        # CSVデータの処理
        if step == 0:
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
            st.markdown("### ダウンロード")
            st.success("✅ 書類作成完了")
            df_result = st.session_state.df_result
            template_path = get_template_config()[selected_template][
                "template_excel_path"
            ]
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
