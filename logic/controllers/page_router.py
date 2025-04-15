import streamlit as st
from config.page_config import page_dict, page_dict_reverse, page_labels
from app_pages.top_page import show_top_page
from app_pages.manage_work import show_manage_work
from components.manual_links import show_manual_links
from components.notice import show_notice
from components.update_log import show_update_log
from components.version_info import show_version_info

def route_page():
    # URLパラメータと初期化
    params = st.query_params
    page_id = params.get("page", "home")
    default_label = page_dict_reverse.get(page_id, "トップページ")

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = default_label

    # サイドバー
    menu = st.sidebar.selectbox("📂 機能を選択", page_labels, key="selected_page")
    st.query_params["page"] = page_dict[st.session_state.selected_page]

    # タイトル
    if st.session_state.selected_page == "トップページ":
        st.title("📘 WEB版 参謀くん")
        show_top_page()
        with st.sidebar:
            st.markdown("---")
            show_notice()
            show_manual_links()
            show_update_log()
            show_version_info()

    elif st.session_state.selected_page == "管理業務":
        st.title("📂 管理業務")
        show_manage_work()

    elif st.session_state.selected_page == "やよい会計":
        st.title("📂 やよい会計")
        st.info("📥 やよい会計インポート機能は現在準備中です。")

    elif st.session_state.selected_page == "機能２":
        st.title("📂 機能２")
        st.info("🧪 新機能２は今後追加予定です。しばらくお待ちください。")
