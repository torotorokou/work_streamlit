# app_pages/manage/page.py
import streamlit as st
from app_pages.base_page import BasePage
from app_pages.manage.controller import manage_work_controller


class ManageWorkPage(BasePage):
    def __init__(self):
        super().__init__(page_id="manage", title="帳票作成")

    def render(self):
        self.render_title()
        st.divider()
        manage_work_controller()  # コントローラ呼び出し
