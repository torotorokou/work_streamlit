import streamlit as st
from utils.config_loader import load_factory_menu_options
from app_pages.factory_manage.pages.balance_management_table.controller import (
    factory_manage_controller,
)
from app_pages.factory_manage.pages.inbound_volume_forecast.render import (
    render_import_volume,
)
from app_pages.factory_manage.view.menu import render_sidebar
from app_pages.factory_manage.view.css import inject_sidebar_css


def factory_manage_work_controller():
    controller = FactoryManageWorkController()
    controller.run()


class FactoryManageWorkController:
    def __init__(self):
        self.menu_options = load_factory_menu_options()
        self.selected_menu_key = None

    def run(self):
        inject_sidebar_css()
        labels = [item["label"] for item in self.menu_options]
        selected_label = render_sidebar(labels)
        self.selected_menu_key = next(
            item["key"] for item in self.menu_options if item["label"] == selected_label
        )
        self.route()

    def route(self):
        if self.selected_menu_key == "inbound_volume":
            render_import_volume()
        elif self.selected_menu_key == "monitor":
            factory_manage_controller()
