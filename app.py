import streamlit as st

# ✅ ページ初期設定
st.set_page_config(page_title="web版 参謀くん", layout="centered")

# 自作モジュールの読み込み
from app_pages.top_page import show_top_page
from components.update_log import show_update_log
from components.manual_links import show_manual_links
from components.notice import show_notice
from components.version_info import show_version_info
from components.ui_style import apply_global_style
from app_pages.manage_work import show_manage_work
from config.page_config import page_dict, page_labels, page_dict_reverse


# ✅ グローバルスタイル適用
apply_global_style()

# ✅ クエリパラメータから現在のページを取得（新方式）
params = st.query_params
page_id = params.get("page", "home")


# ✅ 表示ラベル
page_labels = list(page_dict.keys())
default_label = page_dict_reverse.get(page_id, "トップページ")


# ✅ 初期化：セッションに未設定ならURLから反映
if "selected_page" not in st.session_state:
    st.session_state.selected_page = default_label

# ✅ サイドバーでメニュー選択（session_stateで管理）
menu = st.sidebar.selectbox("📂 機能を選択", page_labels, key="selected_page")

# ✅ URLにも現在ページを反映
st.query_params["page"] = page_dict[st.session_state.selected_page]

# ✅ タイトル表示
if st.session_state.selected_page == "トップページ":
    st.title("📘 WEB版 参謀くん")
else:
    st.title(f"📂 {st.session_state.selected_page}")

# ---------- トップページ ----------
if st.session_state.selected_page == "トップページ":
    # トップページの履歴管理
    if "top_page_viewed" not in st.session_state:
        st.session_state.top_page_viewed = False
    show_top_page()

    with st.sidebar:
        st.markdown("---")
        show_notice()
        st.markdown("---")
        show_manual_links()
        st.markdown("---")
        show_update_log()
        st.markdown("---")
        show_version_info()

# ---------- 管理業務 ----------
elif st.session_state.selected_page == "管理業務":
    show_manage_work()

# ---------- やよい会計 ----------
elif st.session_state.selected_page == "やよい会計":
    st.info("📥 やよい会計インポート機能は現在準備中です。")

# ---------- 機能２ ----------
elif st.session_state.selected_page == "機能２":
    st.info("🧪 新機能２は今後追加予定です。しばらくお待ちください。")
