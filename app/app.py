import streamlit as st
from app_pages.page_router import route_page
from components.ui_style import apply_global_style
from logic.config.yaml_loader.app_setting_yaml import AppSettingLoader
from utils.config_loader import get_app_setting 
from config.env.loader import load_settings


# ページ設定
title = get_app_setting()
st.set_page_config(page_title=title["title"], layout="centered")

# グローバルCSS
apply_global_style()

# ルーティング制御（URLとsession_state）
route_page()


# 開発環境設定
settings = load_settings()
st.title(f"現在の環境: {settings['ENV_NAME']}")
st.write(f"ポート番号: {settings['STREAMLIT_SERVER_PORT']}")
st.write(f"デバッグモード: {settings['DEBUG']}")
