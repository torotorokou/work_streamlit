import streamlit as st
import pandas as pd
import time
from io import BytesIO

# 自作モジュールの読み込み
from app_pages.top_page import show_top_page
from components.update_log import show_update_log
from components.manual_links import show_manual_links
from components.notice import show_notice
from components.version_info import show_version_info
from components.ui_style import apply_global_style
from app_pages.manage_work import show_manage_work
from utils.config_loader import load_config



# ✅ ページ初期設定とスタイル適用
st.set_page_config(page_title="web版 参謀くん", layout="centered")
apply_global_style()
st.query_params["dev_mode"] = "true"

# ✅ サイドバー - メニュー選択
menu = st.sidebar.selectbox("📂 機能を選択", ["トップページ", "管理業務", "機能１", "機能２"])

# ✅ タイトル表示
if menu == "トップページ":
    st.title("📘 WEB版 参謀くん")
else:
    st.title(f"📂 {menu}")

# ---------- トップページ ----------
if menu == "トップページ":
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
elif menu == "管理業務":
     show_manage_work()
