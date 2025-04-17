import streamlit as st
from components.status_box import render_status_box

# from utils.config_loader import (
#     get_template_dict,
#     get_template_descriptions,
#     get_required_files_map,
#     get_csv_label_map,
#     get_csv_date_columns,
#     get_path_config,
# )


def render_manage_page(template_dict, template_descriptions):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 管理業務メニュー")
    template_label = st.sidebar.radio(
        "出力したい項目を選択してください", list(template_dict.keys())
    )

    st.subheader(f"📝 {template_label} の作成")
    description = template_descriptions.get(template_label, "")
    if description:
        st.markdown(
            f"""<div style="margin-left: 2em; color:#ccc;">{description}</div>""",
            unsafe_allow_html=True,
        )

    return template_label  # ✅ 選択結果を返す


def show_upload_status(file):
    if file:
        render_status_box("✅ アップロード済み", "#e6f4ea", "#34a853")
    else:
        render_status_box("⏳ 未アップロード", "#fef7e0", "#f9ab00")


def render_file_upload_section(required_keys, csv_label_map):
    st.markdown("### 📂 CSVファイルのアップロード")
    st.info("以下のファイルをアップロードしてください。")

    uploaded_files = {}
    for key in required_keys:
        label = csv_label_map.get(key, key)
        uploaded_file = st.file_uploader(label, type="csv", key=f"{key}")

        if uploaded_file is not None:
            st.session_state[f"uploaded_{key}"] = uploaded_file
            uploaded_files[key] = uploaded_file
        else:
            uploaded_files[key] = st.session_state.get(f"uploaded_{key}", None)

        show_upload_status(uploaded_files[key])

    return uploaded_files


# app_pages/manage/view.py
def render_status_message(missing_keys, required_keys, uploaded_files, date_columns, selected_template, csv_label_map, config):
    from logic.controllers.csv_controller import prepare_csv_data
    from components.custom_button import centered_button, centered_download_button
    from logic.manage.processor import process_template_to_excel
    from datetime import datetime
    import time

    if not missing_keys:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.markdown("---")

        if centered_button("📊 書類作成"):
            progress = st.progress(0)
            progress.progress(10, "📥 ファイルを処理中...")
            time.sleep(0.3)

            dfs = prepare_csv_data(uploaded_files, date_columns)

            progress.progress(40, "🧮 データを計算中...")
            time.sleep(0.3)

            output_excel = process_template_to_excel(selected_template, dfs, csv_label_map, config)

            progress.progress(90, "✅ 整理完了")
            time.sleep(0.3)
            progress.progress(100)

            today_str = datetime.now().strftime("%Y%m%d")
            st.info("✅ ファイルが生成されました。下のボタンからダウンロードできます👇")

            centered_download_button(
                label="📥 Excelファイルをダウンロード",
                data=output_excel.getvalue(),
                file_name=f"{selected_template}_{today_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        uploaded_count = len(required_keys) - len(missing_keys)
        total_count = len(required_keys)
        st.progress(uploaded_count / total_count)
        st.info(f"📥 {uploaded_count} / {total_count} ファイルがアップロードされました")
