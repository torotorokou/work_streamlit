import streamlit as st
from logic.controllers.page_router import route_page
from components.ui_style import apply_global_style

# ページ設定
st.set_page_config(page_title="web版 参謀くん", layout="centered")

# グローバルCSS
apply_global_style()

# ルーティング制御（URLとsession_state）
route_page()
