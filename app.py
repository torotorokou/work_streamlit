import streamlit as st
from logic.controllers.page_router import route_page
from components.ui_style import apply_global_style
from utils.config_loader import get_app_config


# ページ設定
title = get_app_config()
st.set_page_config(page_title=title['title'], layout="centered")

# グローバルCSS
apply_global_style()

# ルーティング制御（URLとsession_state）
# route_page()
