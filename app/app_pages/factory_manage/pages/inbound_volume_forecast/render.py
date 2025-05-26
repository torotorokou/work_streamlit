import streamlit as st
from datetime import date, timedelta
import pandas as pd
import altair as alt
from logic.factory.predict_model import predict_with_saved_model
from utils.get_holydays import get_japanese_holidays


def render_import_volume():
    st.subheader("🚛 搬入量予測AI（仮）")

    # --- 今週の月〜土を初期値に設定 ---
    today = date.today()
    start = today - timedelta(days=today.weekday())
    end = start + timedelta(days=5)

    # --- カレンダー入力 ---
    selected_dates = st.date_input(
        "📅 予測対象の期間を選択してください", value=(start, end)
    )

    if not (isinstance(selected_dates, tuple) and len(selected_dates) == 2):
        st.warning("⚠️ 2つの日付（開始日と終了日）を選択してください。")
        return

    start_date, end_date = selected_dates
    st.write(f"✅ 選択された期間：{start_date} ～ {end_date}")

    # --- ボタン表示（予測開始トリガー） ---
    if st.button("📌 予測を実行する"):
        holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=True)

        with st.spinner("🤖 AIが予測中..."):
            df_pred = predict_with_saved_model(
                start_date=str(start_date),
                end_date=str(end_date),
                holidays=holidays,
                model_dir="/work/app/data/models",
            )

        st.success("✅ 予測完了！")

        # --- ラベルフィルタ ---
        label_filter = st.multiselect(
            "🔍 表示する判定ラベルを選択してください",
            options=df_pred["判定ラベル"].unique(),
            default=df_pred["判定ラベル"].unique().tolist(),
        )
        df_filtered = df_pred[df_pred["判定ラベル"].isin(label_filter)]

        # --- 強調付きデータ表示 ---
        def highlight_label(val):
            if val == "警告":
                return "background-color: red; color: white"
            elif val == "注意":
                return "background-color: orange; color: black"
            return ""

        st.dataframe(df_filtered.style.applymap(highlight_label, subset=["判定ラベル"]))

        # --- Altair可視化 ---
        chart_data = df_filtered.reset_index().copy()
        chart_data["日付"] = pd.to_datetime(chart_data["日付"])

        line = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(x="日付:T", y="補正後予測:Q", tooltip=["日付:T", "補正後予測:Q"])
        )

        bars = (
            alt.Chart(chart_data)
            .mark_bar(opacity=0.3)
            .encode(x="日付:T", y="補正後予測:Q", color="判定ラベル:N")
        )

        st.altair_chart(line + bars, use_container_width=True)

        # --- CSVダウンロード ---
        csv = df_pred.to_csv().encode("utf-8")
        st.download_button(
            "📥 予測結果をCSVでダウンロード",
            csv,
            file_name="予測結果.csv",
            mime="text/csv",
        )

        # --- セッション保持（任意） ---
        st.session_state["df_import_prediction"] = df_pred
