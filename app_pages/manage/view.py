import streamlit as st
from utils.config_loader import (
    get_template_dict,
    get_template_descriptions,
    get_required_files_map,
    get_csv_label_map,
    get_csv_date_columns,
    get_path_config,
)


def render_manage_page():
    # 変数の宣言
    template_dict = get_template_dict()
    template_descriptions = get_template_descriptions()
    required_files = get_required_files_map()
    csv_label_map = get_csv_label_map()
    date_columns = get_csv_date_columns()

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

    receive_header_definition = get_path_config()["csv"]["receive_header_definition"]
