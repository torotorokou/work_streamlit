import streamlit as st
from app_pages.page_router import route_page
from components.ui_style import apply_global_style
from utils.config_loader import get_app_config
from config.settings.loader import load_settings


# ページ設定
title = get_app_config()
st.set_page_config(page_title=title["title"], layout="centered")

# グローバルCSS
apply_global_style()

# ルーティング制御（URLとsession_state）
route_page()


settings = load_settings()

st.title(f"現在の環境: {settings['ENV_NAME']}")
st.write(f"ポート番号: {settings['STREAMLIT_SERVER_PORT']}")
st.write(f"デバッグモード: {settings['DEBUG']}")