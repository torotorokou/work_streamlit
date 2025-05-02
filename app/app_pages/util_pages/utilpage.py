import streamlit as st
from app_pages.base_page import BasePage


class UtilPage(BasePage):
    def __init__(self):
        super().__init__(page_id="util", title="ユーティリティ機能")

    def render(self):
        self.render_title()
        st.write("各機能を実装予定です")
        st.write("やよい会計インポートなど")
        # 各機能をセクションとして追加
