import streamlit as st
import pandas as pd
import sqlite3
import calendar
from datetime import datetime, timedelta
from utils.config_loader import get_path_from_yaml
from logic.factory_manage.sql import load_recent_dates_from_sql
from logic.factory_manage.style import calendar_header_html, calendar_body_html


def render_calendar_section():
    """
    過去3ヶ月分の搬入データのある日をカレンダー形式で表示する。
    データがある日は緑色でハイライトされる。

    Returns:
        None
    """
    sql_url = get_path_from_yaml("weight_data", section="sql_database")
    dates_with_data = load_recent_dates_from_sql(sql_url)
    months = [datetime.today() - pd.DateOffset(months=i) for i in range(2, -1, -1)]
    cols = st.columns(3)

    for i, month in enumerate(months):
        with cols[i]:
            html = generate_calendar_html(month.year, month.month, dates_with_data)
            st.markdown(html, unsafe_allow_html=True)


def generate_calendar_html(
    year: int, month: int, highlight_dates: list[datetime.date]
) -> str:
    """
    指定された年月のHTMLカレンダーを生成する。

    Args:
        year (int): 対象の年（例: 2025）
        month (int): 対象の月（例: 5）
        highlight_dates (list[date]): ハイライトする日付リスト

    Returns:
        str: HTMLカレンダーの文字列
    """
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdayscalendar(year, month)

    html = calendar_header_html(year, month)
    html += calendar_table_header_row()
    html += calendar_body_html(weeks, year, month, highlight_dates)
    html += "</table></div><br>"

    return html


def calendar_table_header_row() -> str:
    """
    カレンダーの曜日ヘッダー行をHTMLとして生成。

    Returns:
        str: 曜日ヘッダーのHTML
    """
    weekdays = ["日", "月", "火", "水", "木", "金", "土"]
    weekday_colors = ["#d9534f", "#333", "#333", "#333", "#333", "#333", "#0275d8"]

    row = "<tr>"
    for i, day in enumerate(weekdays):
        row += f"<th style='padding: 6px; color: {weekday_colors[i]}; font-weight: 500;'>{day}</th>"
    row += "</tr>"
    return row
