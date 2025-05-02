import streamlit as st
from utils.logger import app_logger
from utils.config_loader import get_template_config
from abc import ABC, abstractmethod


class BasePage(ABC):
    def __init__(self, page_id: str, title: str = ""):
        self.page_id = page_id
        self.config = get_template_config().get(page_id, {})
        self.title = title or self.config.get("title", page_id)
        self.logger = app_logger()

    def render_title(self):
        st.title(f"📄 {self.title}")

    def log(self, message: str):
        self.logger.info(f"[{self.page_id}] {message}")

    def render_centered(self, render_fn):
        col1, col2, col3 = st.columns([2, 6, 2])
        with col2:
            render_fn()

    @abstractmethod
    def render(self):
        """サブクラスでページ描画を実装する必要があります"""
        pass
