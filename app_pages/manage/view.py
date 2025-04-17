import streamlit as st
from components.status_box import render_status_box
from components.custom_button import centered_download_button
from io import BytesIO

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
    all_keys = list(csv_label_map.keys())

    for key in all_keys:
        label = csv_label_map.get(key, key)

        # 表示ON or OFF を条件で分岐
        if key in required_keys:
            # 通常のアップロードUI
            uploaded_file = st.file_uploader(label, type="csv", key=f"{key}")
        else:
            # 👇 session_stateを維持するために「非表示」でfile_uploaderを実行
            uploaded_file = st.file_uploader(
                label,
                type="csv",
                key=f"{key}",
                disabled=True,
                label_visibility="collapsed"  # ← これで見えなくなるけど内部的には維持される！
            )

        # 共通：アップロードされたら session_state に保存
        if uploaded_file is not None:
            st.session_state[f"uploaded_{key}"] = uploaded_file
            uploaded_files[key] = uploaded_file
        else:
            # 表示されているCSVのみ削除（非表示のCSVは保持）
            if key in required_keys:
                if f"uploaded_{key}" in st.session_state:
                    del st.session_state[f"uploaded_{key}"]
                uploaded_files[key] = None
            else:
                # 非表示のものは session_state に残っていればそのまま使う
                uploaded_files[key] = st.session_state.get(f"uploaded_{key}", None)

        # ステータスは必要なCSVのみ表示
        if key in required_keys:
            show_upload_status(uploaded_files[key])

    return uploaded_files




# app_pages/manage/view.py
def render_status_message_ui(
    file_ready: bool,
    file_name: str = None,
    output_excel: BytesIO = None,
    uploaded_count: int = 0,
    total_count: int = 0,
):

    if file_ready and output_excel:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.info("✅ ファイルが生成されました。下のボタンからダウンロードできます👇")
        centered_download_button(
            label="📥 Excelファイルをダウンロード",
            data=output_excel.getvalue(),
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.progress(uploaded_count / total_count)
        st.info(f"📥 {uploaded_count} / {total_count} ファイルがアップロードされました")
