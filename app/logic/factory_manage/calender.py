import streamlit as st
import pandas as pd
import sqlite3
import calendar
from datetime import datetime, timedelta

from utils.config_loader import get_path_from_yaml
from logic.factory_manage.sql import load_recent_dates_from_sql


# --- カレンダー表示（過去3ヶ月分） ---
def render_calendar_section():
    sql_url = get_path_from_yaml("weight_data", section="sql_database")
    dates_with_data = load_recent_dates_from_sql(sql_url)

    months = [datetime.today() - pd.DateOffset(months=i) for i in range(2, -1, -1)]
    cols = st.columns(3)

    for i, month in enumerate(months):
        with cols[i]:
            html = generate_calendar_html(month.year, month.month, dates_with_data)
            st.markdown(html, unsafe_allow_html=True)


# --- カレンダーHTMLを生成（スタイリッシュ版） ---
def generate_calendar_html(year: int, month: int, highlight_dates):
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdayscalendar(year, month)
    html = f"""
    <div style='
        background-color: var(--background);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 0 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0 auto;
        color: var(--text);
    '>
    <h4 style='margin-bottom: 0.5rem; font-weight: 600;'>{year}年 {month}月</h4>
    <table style='border-collapse: collapse; margin: 0 auto; font-size: 13px;'>
        <tr>
    """

    weekdays = ["日", "月", "火", "水", "木", "金", "土"]
    weekday_colors = ["#d9534f", "#333", "#333", "#333", "#333", "#333", "#0275d8"]

    # ヘッダー
    for i, day in enumerate(weekdays):
        html += f"<th style='padding: 6px; color: {weekday_colors[i]}; font-weight: 500;'>{day}</th>"
    html += "</tr>"

    # 日付セル
    for week in weeks:
        html += "<tr>"
        for i, day in enumerate(week):
            if day == 0:
                html += "<td style='padding: 6px;'></td>"
            else:
                date_obj = datetime(year, month, day).date()
                has_data = date_obj in highlight_dates
                bg = "#90ee90" if has_data else "#f5f5f5"
                text_color = "#222" if has_data else "#999"
                html += f"""
                <td style='
                    padding: 6px;
                    text-align: center;
                    background-color: {bg};
                    color: {text_color};
                    border-radius: 8px;
                    border: 1px solid #ddd;
                    width: 36px;
                    height: 36px;
                    font-weight: 500;
                '>{day}</td>"""
        html += "</tr>"
    html += "</table></div><br>"
    return html
