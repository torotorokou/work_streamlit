# app_pages/manage/page.py

from app_pages.base_page import BasePage
from app_pages.manage.controller import manage_work_controller


class ManageWorkPage(BasePage):
    def __init__(self):
        super().__init__(page_id="manage", title="管理業務")

    def render(self):
        self.render_title()
        manage_work_controller()  # コントローラ呼び出し
