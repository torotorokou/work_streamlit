import streamlit as st
from app_pages.manage_work import show_manage_work
from components.ui_style import apply_global_style

st.set_page_config(page_title="管理業務 | WEB版 参謀くん", layout="centered")
apply_global_style()

st.title("📂 管理業務")

show_manage_work()