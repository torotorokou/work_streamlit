import streamlit as st
from utils.config_loader import (
    get_csv_date_columns,
    get_csv_label_map,
    get_required_files_map,
    get_template_descriptions,
    get_template_dict,
    get_path_config,
)


from app_pages.manage.view import render_manage_page


def manage_work_controller():
    # 最初のUI表示
    render_manage_page()
