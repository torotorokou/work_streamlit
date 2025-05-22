import streamlit as st
from components.status_box import render_status_box
from components.custom_button import centered_download_button
from components.ui_message import show_warning_bubble
from logic.detect_csv import detect_csv_type
from io import BytesIO
from typing import Optional
from utils.config_loader import get_csv_label_map


def render_manage_page(template_dict, template_descriptions):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 帳票作成メニュー")
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
    if not file:
        render_status_box(
            message="  未アップロード",
            bg_rgba="rgba(244, 67, 54, 0.07)",  # やや赤みのある背景
            text_color="#e57373",  # 明るめの赤
        )


def render_upload_header(title: str):
    st.markdown(
        f"""
    <div style="
        background-color: rgba(255, 223, 89, 0.15);  /* 上品な薄黄色 */
        color: #222;
        padding: 10px 16px;
        margin-top: 24px;
        margin-bottom: 10px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 15px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span style="font-size: 17px;">📁</span>
        <span>{title}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


import tempfile


def render_file_upload_section(required_keys):
    csv_label_map = get_csv_label_map()
    uploaded_files = {}
    all_keys = list(csv_label_map.keys())

    for key in all_keys:
        label = csv_label_map.get(key, key)

        # --- 必要なCSVファイル（通常表示） ---
        if key in required_keys:
            render_upload_header(label)
            uploaded_file = st.file_uploader(
                label, type="csv", key=f"{key}", label_visibility="collapsed"
            )

            if uploaded_file is not None:
                try:
                    # ✅ tempfile に書き込み（日本語ファイル名問題回避）
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".csv"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    expected_name = label
                    detected_name = detect_csv_type(
                        tmp_path
                    )  # ← 関数側もパス受け取りに変更すること
                    if detected_name != expected_name:
                        show_warning_bubble(expected_name, detected_name)
                        st.session_state[f"uploaded_{key}"] = None
                        uploaded_files[key] = None
                    else:
                        st.session_state[f"uploaded_{key}"] = tmp_path
                        uploaded_files[key] = tmp_path
                except Exception as e:
                    st.error(f"ファイルの保存または検出に失敗しました: {e}")
                    uploaded_files[key] = None
            else:
                if f"uploaded_{key}" in st.session_state:
                    del st.session_state[f"uploaded_{key}"]
                uploaded_files[key] = None

            show_upload_status(uploaded_files[key])

        # --- 不要なCSVファイル（保持） ---
        else:
            with st.expander(
                f"🗂 {label}（このテンプレートでは不要です）", expanded=False
            ):
                st.caption(
                    "このファイルは他のテンプレートで使用されます。削除する必要はありません。"
                )
                uploaded_file = st.file_uploader(
                    label,
                    type="csv",
                    key=f"{key}",
                    disabled=True,
                    label_visibility="collapsed",
                )
                if uploaded_file is not None:
                    st.session_state[f"uploaded_{key}"] = uploaded_file
                    uploaded_files[key] = uploaded_file
                else:
                    uploaded_files[key] = st.session_state.get(f"uploaded_{key}", None)

    return uploaded_files


# app_pages/manage/view.py
def render_status_message_ui(
    file_ready: bool,
    file_name: Optional[str] = None,
    output_excel: Optional[BytesIO] = None,
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
