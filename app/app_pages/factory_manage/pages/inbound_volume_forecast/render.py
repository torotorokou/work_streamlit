import streamlit as st
from datetime import date, timedelta
import pandas as pd
import altair as alt
from logic.factory.predict_model import predict_with_saved_model
from utils.get_holydays import get_japanese_holidays


def render_import_volume():
    st.title("📊 搬入量予測(仮)")
    st.markdown(
        """予測したい期間を選択して下さい。  
    デフォルトは今週の月曜日から土曜日までです。"""
    )

    # --- 期間選択 ---
    today = date.today()
    default_start = today - timedelta(days=today.weekday())
    default_end = default_start + timedelta(days=5)
    selected_dates = st.date_input("期間を選択", value=(default_start, default_end))

    if not (isinstance(selected_dates, tuple) and len(selected_dates) == 2):
        st.info("開始日と終了日を選択してください。")
        return

    start_date, end_date = selected_dates
    st.caption(f"対象期間: {start_date} ～ {end_date}")

    # --- 予測ボタン（中央配置） ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_prediction = st.button("予測を実行する")

    if run_prediction:
        holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=True)

        with st.spinner("予測中..."):
            df_pred = predict_with_saved_model(
                start_date=str(start_date),
                end_date=str(end_date),
                holidays=holidays,
                model_dir="/work/app/data/models",
            )
        st.session_state["df_import_prediction"] = df_pred
        st.success("予測が完了しました。")

    # --- 結果表示 ---
    if "df_import_prediction" in st.session_state:
        df_pred = st.session_state["df_import_prediction"]

        # ラベルフィルタ
        label_filter = st.multiselect(
            "表示するラベル",
            options=df_pred["判定ラベル"].unique(),
            default=df_pred["判定ラベル"].unique().tolist(),
        )
        df_filtered = df_pred[df_pred["判定ラベル"].isin(label_filter)]

        # 整形処理
        df_display = df_filtered.copy()
        df_display["曜日"] = pd.to_datetime(df_display.index).weekday.map(
            lambda x: "月火水木金土日"[x]
        )
        for col in ["予測値", "補正後予測", "下限95CI", "上限95CI"]:
            df_display[col] = df_display[col].round(0).astype(int)
        df_display["未満確率"] = df_display["未満確率"].apply(
            lambda x: f"{float(x) * 100:.1f}%" if pd.notnull(x) else ""
        )
        df_display["日付"] = df_display.index  # 明示的な列として日付追加

        # 表示用（インデックス除外して表示）
        df_display_show = df_display[
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

        def style_label(val):
            if val == "警告":
                return "color: red; font-weight: bold"
            elif val == "注意":
                return "color: orange"
            return ""

        st.dataframe(df_display_show.style.applymap(style_label, subset=["判定ラベル"]))

        # --- グラフ表示（Altair） ---
        chart_data = df_display.copy()
        chart_data["日付"] = pd.to_datetime(chart_data["日付"])
        chart_data["日付_str"] = chart_data["日付"].dt.strftime("%m/%d")

        base = alt.Chart(chart_data).encode(x=alt.X("日付_str:N", title="日付"))

        error_bars = base.mark_rule(color="green").encode(
            y="下限95CI:Q", y2="上限95CI:Q"
        )

        line = base.mark_line(point=True).encode(
            y=alt.Y(
                "補正後予測:Q",
                title="補正後予測",
                scale=alt.Scale(domain=[50000, 100000]),
            ),
            color=alt.value("#4c78a8"),
            tooltip=["日付_str:N", "補正後予測:Q", "判定ラベル:N"],
        )

        st.altair_chart(
            (error_bars + line).properties(height=300), use_container_width=True
        )

        # --- CSVダウンロード（中央配置＋黄色グラデーション + Shift_JIS） ---
        df_download = df_display[
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

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <style>
                div[data-testid="stDownloadButton"] > button {
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

            # ファイル名に期間を含める（例：20240526_20240531_予測結果.csv）
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            filename = f"{start_str}_{end_str}_予測結果.csv"

            st.download_button(
                label="📥 CSVダウンロード",
                data=csv,
                file_name=filename,
                mime="text/csv; charset=shift_jis",
            )
