from app_pages.base_page import BasePage
from utils.page_config import PageConfig
import streamlit as st


class UtilPage(BasePage):
    def __init__(self):
        config = PageConfig(
            page_id="util",
            title="ユーティリティ機能",
            parent_title="補助ツール"
        )
        super().__init__(config)

    def render(self):
        self.render_title()
        st.write("各機能を実装予定です")
        st.write("やよい会計インポートなど")
        # 各機能をセクションとして追加
