# ✅ 標準ライブラリ

# ✅ サードパーティ
import streamlit as st

# ✅ プロジェクト内 - components（UI共通パーツ）

# ✅ プロジェクト内 - view（UIビュー）
from app_pages.manage.view import (
    render_file_upload_section,
    render_manage_page,
)

# ✅ プロジェクト内 - logic（処理・データ変換など）
from logic.manage.utils.upload_handler import handle_uploaded_files

# ✅ プロジェクト内 - utils（共通ユーティリティ）
from utils.config_loader import (
    get_csv_date_columns,
    get_csv_label_map,
    get_required_files_map,
    get_template_descriptions,
    get_template_dict,
    get_template_config,
)


def load_work_config():
    return {
        "template_dict": get_template_dict(),
        "template_descriptions": get_template_descriptions(),
        "required_files": get_required_files_map(),
        "csv_label_map": get_csv_label_map(),
        "date_columns": get_csv_date_columns(),
        "template_config": get_template_config(),
    }


def select_template_ui(template_dict, template_descriptions):
    selected_label = render_manage_page(template_dict, template_descriptions)
    return selected_label, template_dict[selected_label]


def handle_template_change(current_template):
    if "selected_template_cache" not in st.session_state:
        st.session_state.selected_template_cache = current_template
    elif st.session_state.selected_template_cache != current_template:
        st.session_state.process_step = None
        st.session_state.df_result = None
        st.session_state.extracted_date = None
        st.session_state.selected_template_cache = current_template


def show_upload_ui(required_keys, csv_label_map):
    st.markdown("### 📂 CSVファイルのアップロード")
    st.info("以下のファイルをアップロードしてください。")
    uploaded = render_file_upload_section(required_keys, csv_label_map)
    validated = handle_uploaded_files(required_keys, csv_label_map)
    return uploaded, validated
