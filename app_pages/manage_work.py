# app_pages/manage_work.py
import streamlit as st
import time
from datetime import datetime
from logic.detect_csv import detect_csv_type
from utils.config_loader import load_config
from components.ui_message import show_warning_bubble
from logic.eigyo_management import template_processors
from components.custom_button import centered_button, centered_download_button
from logic.controllers.csv_controller import prepare_csv_data
from utils.debug_tools import save_debug_parquets
from utils.write_excel import write_values_to_template



def show_manage_work():
    # --- 内部データ定義 ---
    template_dict = {
        "工場日報": "factory_report",
        "工場搬出入収支表": "balance_sheet",
        "集計項目平均表": "average_sheet",
        "管理票": "management_sheet",
    }

    template_descriptions = {
        "工場日報": "ヤードと出荷データをもとに、工場内の稼働日報を出力します。",
        "工場搬出入収支表": "受入・ヤード・出荷一覧から収支表を自動集計します。",
        "集計項目平均表": "受入データをABC分類し、各品目の平均値を算出して出力します。",
        "管理票": "受入・ヤード・出荷の一覧を使って管理用の帳票を出力します。",
    }

    required_files = {
        "factory_report": ["yard", "shipping"],
        "balance_sheet": ["receive", "yard", "shipping"],
        "average_sheet": ["receive"],
        "management_sheet": ["receive", "yard", "shipping"],
    }

    csv_label_map = {
        "yard": "ヤード一覧",
        "shipping": "出荷一覧",
        "receive": "受入一覧",
    }

    date_columns = {"receive": "伝票日付", "yard": "伝票日付", "shipping": "伝票日付"}

    # --- UI ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 管理業務メニュー")
    template_label = st.sidebar.radio(
        "出力したい項目を選択してください", list(template_dict.keys())
    )
    selected_template = template_dict.get(template_label)
    uploaded_files = {}

    st.subheader(f"📝 {template_label} の作成")
    description = template_descriptions.get(template_label, "")
    if description:
        st.markdown(
            f"""<div style=\"margin-left: 2em; color:#ccc;\">{description}</div>""",
            unsafe_allow_html=True,
        )

    config = load_config()
    header_csv_path = config["main_paths"]["receive_header_definition"]

    # --- ヘッダーCSVのアップロード ---
    with st.container():
        st.markdown("### 📂 CSVファイルのアップロード")
        st.info("以下のファイルをアップロードしてください。")

        for file_key in required_files[selected_template]:
            label = csv_label_map.get(file_key, file_key)
            uploaded_file = st.file_uploader(
                label, type="csv", key=f"{file_key}_{selected_template}"
            )

            # ✅ 新しいアップロードがあればセッションに保存
            if uploaded_file is not None:
                st.session_state[f"uploaded_{file_key}"] = uploaded_file
                uploaded_files[file_key] = uploaded_file
            else:
                uploaded_files[file_key] = st.session_state.get(f"uploaded_{file_key}", None)

            # 🔍 自動判別チェック
            if uploaded_files[file_key] is not None:
                detected_name = detect_csv_type(uploaded_files[file_key], header_csv_path)
                expected_name = label
                if detected_name != expected_name:
                    show_warning_bubble(expected_name, detected_name)
                    uploaded_files[file_key] = None
                    st.session_state[f"uploaded_{file_key}"] = None

            # ✅ アップロード状態表示（file_uploaderの直下）
            if uploaded_files.get(file_key):
                st.markdown(
                    """
                    <div style="margin-top: -0.5em; margin-bottom: 1.5em; padding: 0.4em 1em;
                                background-color: #e6f4ea; border-left: 4px solid #34a853;
                                border-radius: 4px; font-weight: 500; color: #111;">
                        ✅ アップロード済み
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="margin-top: -0.5em; margin-bottom: 1.5em; padding: 0.4em 1em;
                                background-color: #fef7e0; border-left: 4px solid #f9ab00;
                                border-radius: 4px; font-weight: 500; color: #111;">
                        ⏳ 未アップロード
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # --- ファイルチェック ---
    required_keys = required_files[selected_template]
    missing_keys = [k for k in required_keys if uploaded_files.get(k) is None]

    # --- ステータス表示 ---
    if not missing_keys:
        st.success("✅ 必要なファイルがすべてアップロードされました！")

        st.markdown("---")
        if centered_button("📊 書類作成"):
            progress = st.progress(0)

            progress.progress(10, "📥 ファイルを処理中...")
            time.sleep(0.3)
            dfs = prepare_csv_data(uploaded_files, date_columns)

            processor_func = template_processors.get(selected_template)
            if processor_func:
                progress.progress(40, "🧮 データを計算中...")
                time.sleep(0.3)
                df = processor_func(dfs, csv_label_map)

                progress.progress(70, "📄 テンプレートに書き込み中...")
                time.sleep(0.3)
                template_path = config["templates"][selected_template]["template_excel_path"]
                output_excel = write_values_to_template(df, template_path)

                progress.progress(90, "✅ 整理完了")
                time.sleep(0.3)

                progress.progress(100)
                today_str = datetime.now().strftime("%Y%m%d")

                st.info("✅ ファイルが生成されました。下のボタンからダウンロードできます👇")

                centered_download_button(
                    label="📥 Excelファイルをダウンロード",
                    data=output_excel.getvalue(),
                    file_name=f"{template_label}_{today_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    else:

        uploaded_count = len(required_keys) - len(missing_keys)
        total_count = len(required_keys)

        st.progress(uploaded_count / total_count)
        st.info(f"📥 {uploaded_count} / {total_count} ファイルがアップロードされました")
