import streamlit as st
from app_pages.page_router import route_page
from components.ui_style import apply_global_style
from utils.config_loader import get_app_config
from config.settings.loader import load_settings
from utils.logger import app_logger


# ページ設定
title = get_app_config()
st.set_page_config(page_title=title["title"], layout="centered")

# グローバルCSS
apply_global_style()

# ルーティング制御（URLとsession_state）
route_page()


# 開発環境設定
settings = load_settings()
# logger = app_logger()
# logger.info(f"現在の環境: {settings['ENV_NAME']}")
# logger.info(f"ポート番号: {settings['STREAMLIT_SERVER_PORT']}")
# logger.info(f"デバッグモード: {settings['DEBUG']}")
