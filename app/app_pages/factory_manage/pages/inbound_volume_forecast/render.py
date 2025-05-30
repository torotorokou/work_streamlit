import streamlit as st
from datetime import date, timedelta, datetime
import pandas as pd
import altair as alt
import sqlite3

from logic.factory_manage.calender import (
    generate_calendar_html,
)
from utils.config_loader import get_path_from_yaml
from app_pages.factory_manage.pages.inbound_volume_forecast.controller import (
    csv_controller,
    predict_hannyu_ryou_controller,
)
from logic.factory_manage.style import style_label
from logic.factory_manage.calender import render_calendar_section
from logic.factory_manage.sql import load_recent_dates_from_sql


# --- AIモデルを用いた予測実行処理 ---
def render_prediction_section(start_date, end_date):
    with st.spinner("予測中..."):
        # 旧バージョンで、一括でモデル構築・予測
        df_pred = predict_hannyu_ryou_controller(
            start_date=str(start_date), end_date=str(end_date)
        )
    df_pred = df_pred.copy()
    df_pred["曜日"] = pd.to_datetime(df_pred.index).weekday.map(
        lambda x: "月火水木金土日"[x]
    )
    df_pred["日付"] = df_pred.index
    st.session_state["df_import_prediction"] = df_pred
    st.success("予測が完了しました。")


# --- 表示テーブルとAltairチャート（棒グラフ）を描画 ---
def render_prediction_table_and_chart():
    df_pred = st.session_state["df_import_prediction"]

    label_filter = st.multiselect(
        "表示するラベル",
        options=df_pred["判定ラベル"].unique(),
        default=list(df_pred["判定ラベル"].unique()),
    )
    df_filtered = df_pred[df_pred["判定ラベル"].isin(label_filter)]

    df_display = df_filtered.copy()
    for col in ["予測値", "補正後予測", "下限95CI", "上限95CI"]:
        df_display[col] = df_display[col].round(0).astype(int)
    df_display["未満確率"] = df_display["未満確率"].apply(
        lambda x: f"{float(x) * 100:.1f}%" if pd.notnull(x) else ""
    )

    df_show = df_display[
        [
            "日付",
            "曜日",
            "予測値",
            "補正後予測",
            "下限95CI",
            "上限95CI",
            "判定ラベル",
            "未満確率",
        ]
    ].reset_index(drop=True)
    st.dataframe(df_show.style.applymap(style_label, subset=["判定ラベル"]))

    chart_data = df_display.copy()
    chart_data["日付"] = pd.to_datetime(chart_data["日付"])
    chart_data["日付_str"] = chart_data["日付"].dt.strftime("%m/%d")

    y_max = chart_data["補正後予測"].max()
    y_buffer = int(y_max * 0.1)

    # --- 棒グラフ描画（色：ラベル連動、枠：青） ---
    bar = (
        alt.Chart(chart_data)
        .mark_bar(size=30, stroke="blue", strokeWidth=1)
        .encode(
            x=alt.X("日付_str:N", title="日付"),
            y=alt.Y(
                "補正後予測:Q",
                title="補正後予測",
                scale=alt.Scale(domain=[0, y_max + y_buffer]),
            ),
            color=alt.Color(
                "判定ラベル:N",
                scale=alt.Scale(
                    domain=["警告", "注意", "通常"], range=["red", "orange", "#4c78a8"]
                ),
                legend=None,
            ),
            tooltip=["日付_str:N", "補正後予測:Q", "判定ラベル:N"],
        )
    )
    st.altair_chart(bar.properties(height=400), use_container_width=True)


# --- CSVダウンロードボタンの表示処理 ---
def render_download_button(start_date, end_date):
    df = st.session_state["df_import_prediction"]
    df_download = df[
        [
            "日付",
            "曜日",
            "予測値",
            "補正後予測",
            "下限95CI",
            "上限95CI",
            "判定ラベル",
            "未満確率",
        ]
    ].copy()
    csv = df_download.to_csv(index=False, encoding="shift_jis")

    # --- ダウンロードボタンのCSS装飾 ---
    st.markdown(
        """
        <style>
        div[data-testid=\"stDownloadButton\"] > button {
            background: linear-gradient(to right, #fddb3a, #f6b93b);
            color: black;
            border: none;
            font-weight: bold;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    filename = (
        f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_予測結果.csv"
    )
    st.download_button(
        "📥 CSVダウンロード",
        data=csv,
        file_name=filename,
        mime="text/csv; charset=shift_jis",
    )


# --- アプリ全体のメインUI関数 ---
def render_import_volume():
    # ページタイトルと説明
    st.title("📊 搬入量予測AI")

    # --- カレンダー表示 ---
    st.subheader("📅 読込済CSV日付")
    st.markdown(
        """現在読込済みのCSV一覧表です。  
    追加する場合は、以下からCSVをアップロードして下さい。"""
    )
    render_calendar_section()

    # --- CSVのアップロード（折りたたみ式） ---
    with st.expander("📂 CSVのアップロードはこちらをクリック", expanded=False):
        st.markdown("""追加したいCSVをアップロードして下さい。""")
        csv_controller()

    # --- 日付選択UI（週の月曜〜土曜）を初期値に設定 ---
    st.subheader("📅 予測期間の選択")
    st.markdown(
        """予測したい期間を選択して下さい。  
    デフォルトは今週の月曜日から土曜日までです。"""
    )

    today = date.today()
    default_start = today - timedelta(days=today.weekday())  # 月曜
    default_end = default_start + timedelta(days=5)  # 土曜
    selected_dates = st.date_input("期間を選択", value=(default_start, default_end))

    # --- バリデーション: 日付が2つとも指定されているか確認 ---
    if not (isinstance(selected_dates, tuple) and len(selected_dates) == 2):
        st.info("開始日と終了日を選択してください。")
        return

    # --- 入力された期間を変数に代入 ---
    start_date, end_date = selected_dates
    # st.caption(f"対象期間: {start_date} ～ {end_date}")

    # --- 予測実行ボタン（中央配置） ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("予測を実行する"):
            render_prediction_section(start_date, end_date)

    # --- 予測結果が存在する場合のみ、表とチャート、ダウンロードボタンを表示 ---
    if "df_import_prediction" in st.session_state:
        render_prediction_table_and_chart()
        render_download_button(start_date, end_date)
