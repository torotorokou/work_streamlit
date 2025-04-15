import streamlit as st
# from config.page_config import page_dict, page_dict_reverse, page_labels
from utils.config_loader import get_page_dicts
from app_pages.top_page import show_top_page
from app_pages.manage_work import show_manage_work
from components.manual_links import show_manual_links
from components.notice import show_notice
from components.update_log import show_update_log
from components.version_info import show_version_info


# controller: route_page.py

def route_page():
    """
    Streamlitアプリのルーティング処理を行うメイン関数。

    - URLクエリパラメータとセッション状態を同期し、
    - サイドバーにページメニューを表示、
    - 選択されたページの中身を描画する。

    ページ構成情報（ID・ラベル）は YAML から読み込み、 
    MVC構成のController的役割を担う。
    """
    # ページ構成情報を取得（ラベルとURL ID）
    page_dict, page_dict_reverse, page_labels = get_page_dicts()

    # URLパラメータとセッションの同期
    _handle_query_params(page_dict, page_dict_reverse, page_labels)

    # サイドバーに選択メニュー表示
    _render_sidebar(page_labels)

    # 選択されたページの中身を描画
    _render_selected_page()


# ↓↓↓↓ 以下は内部関数へ ↓↓↓↓

def _handle_query_params(page_dict, page_dict_reverse):
    params = st.query_params
    page_id = params.get("page", "home")
    default_label = page_dict_reverse.get(page_id, "トップページ")

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = default_label
    
    st.query_params["page"] = page_dict[st.session_state.selected_page]


def _render_sidebar(page_labels):
    st.sidebar.selectbox("📂 機能を選択", page_labels, key="selected_page")


def _render_selected_page():
    selected = st.session_state.selected_page

    if selected == "トップページ":
        st.title("📘 WEB版 参謀くん")
        show_top_page()
        _render_sidebar_addons()
    elif selected == "管理業務":
        st.title("📂 管理業務")
        show_manage_work()
    elif selected == "やよい会計":
        st.title("📂 やよい会計")
        st.info("📥 やよい会計インポート機能は現在準備中です。")
    elif selected == "機能２":
        st.title("📂 機能２")
        st.info("🧪 新機能２は今後追加予定です。しばらくお待ちください。")


def _render_sidebar_addons():
    with st.sidebar:
        st.markdown("---")
        show_notice()
        show_manual_links()
        show_update_log()
        show_version_info()
