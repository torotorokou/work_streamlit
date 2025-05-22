import streamlit as st

# ✅ 標準ライブラリ
import time

# ✅ サードパーティ
import streamlit as st
import pandas as pd

# ✅ プロジェクト内 - components（UI共通パーツ）
from components.custom_button import centered_button, centered_download_button
from components.custom_progress_bar import CustomProgressBar

# ✅ プロジェクト内 - view（UIビュー）
from app_pages.manage.view import (
    render_file_upload_section,
    render_manage_page,
)

# ✅ プロジェクト内 - logic（処理・データ変換など）
from logic.manage import template_processors
from logic.controllers.csv_controller import prepare_csv_data
from logic.manage.utils.upload_handler import handle_uploaded_files
from logic.manage.utils.file_validator import check_missing_files

# ✅ プロジェクト内 - utils（共通ユーティリティ）
from utils.progress_helper import update_progress
from utils.logger import app_logger
from utils.write_excel import write_values_to_template
from utils.debug_tools import save_debug_parquets
from utils.config_loader import (
    get_csv_date_columns,
    get_csv_label_map,
    get_required_files_map,
    get_template_descriptions,
    get_template_dict,
    get_template_config,
)

from utils.config_loader import load_factory_required_files
from utils.config_loader import get_csv_label_map
from utils.file_loader import load_uploaded_csv_files


def factory_manage_controller():
    st.subheader("🗑 工場収支モニタリング表")
    st.write("処理実績や分類別の集計を表示します。")

    selected_template = "monitor"
    # --- 必要ファイルキーを取得 ---
    required_keys = load_factory_required_files()[selected_template]

    # 🔽 再計算用
    if "selected_template_cache" not in st.session_state:
        st.session_state.selected_template_cache = selected_template
    elif st.session_state.selected_template_cache != selected_template:
        st.session_state.process_step = None
        st.session_state.df_result = None
        st.session_state.extracted_date = None
        st.session_state.selected_template_cache = selected_template

    # --- ファイルアップロードUI表示 & 取得 ---
    render_shipping_upload_section()

    # --- 整合性チェック（session_stateから取得 → validate）
    uploaded_files = handle_uploaded_files(required_keys)

    # --- アップロード状態チェック（単一ファイル）
    uploaded_file = uploaded_files.get("shipping")
    all_uploaded, missing_key = check_single_file_uploaded(uploaded_file, "shipping")
    print(all_uploaded, missing_key)

    # --- セッション初期化（ファイルが足りないとき）
    if not all_uploaded and "process_step" in st.session_state:
        st.session_state.process_step = None
        st.session_state.dfs = None
        st.session_state.df_result = None
        st.session_state.extracted_date = None

    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.markdown("---")

        # --- ステップ管理の初期化（ボタン押下前は None）---
        if "process_step" not in st.session_state:
            st.session_state.process_step = None

        # --- 書類作成ボタンを最初に表示し、押されたらステップ0へ移行 ---
        if st.session_state.process_step is None:
            if centered_button("⏩ 書類作成を開始する"):
                st.session_state.process_step = 0
                st.rerun()
            return  # ボタンが押されるまでは処理しない

        # --- ステップ制御とプログレス描画 ---
        step = st.session_state.get("process_step", 0)
        progress_bar = CustomProgressBar(
            total_steps=3, labels=["📥 読込中", "🧮 処理中", "📄 出力"]
        )

        # ✅ プログレスバーの描画
        with st.container():
            progress_bar.render(step)

        # CSVデータの処理 毎月になっているか。
        if step == 0:
            # --- CSV読み込み ---
            df = pd.read_csv(uploaded_file)

            # --- (曜日) の除去 → 文字列 → 日付に変換 ---
            df["伝票日付"] = df["伝票日付"].str.replace(r"\s*\([^)]+\)", "", regex=True)
            df["伝票日付"] = pd.to_datetime(
                df["伝票日付"], errors="coerce", format="mixed"
            )

            # --- 日付変換失敗時はエラー ---
            if df["伝票日付"].isna().all():
                st.error(
                    "📛 日付列のパースに失敗しました。列名やフォーマットを確認してください。"
                )
                st.stop()

            # --- 並び替え ---
            df = df.sort_values(by="伝票日付", ascending=True)

            # --- 最初の日付から「年・月」を抽出 ---
            first_date = df["伝票日付"].iloc[0]
            target_year = first_date.year
            target_month = first_date.month

            # --- 月が一致しない行を除外 ---
            df = df[
                (df["伝票日付"].dt.year == target_year)
                & (df["伝票日付"].dt.month == target_month)
            ]

            # --- 常に表示したいメッセージ（セッションに保存して後でも表示可能） ---
            message = f"{target_year}年{target_month}月の現在数量を作成します"
            st.session_state.message = message  # ← セッションに保存しておく

            # --- 表示 ---
            st.markdown(f"### 📅 {message}")

            # --- データ保存＆ステップ進行 ---
            dfs = {"shipping": df}
            save_debug_parquets(dfs)
            st.session_state.dfs = dfs
            st.session_state.extracted_date = first_date.strftime("%Y%m%d")
            st.session_state.process_step = 1
            st.rerun()

        elif step == 1:
            # 日付の表示
            st.markdown(f"### 📅 {st.session_state.message}")

            # 詳細の処理
            df_result = processor_func(st.session_state.dfs)

            if df_result is None:
                st.stop()  # UI選択画面などで中断されている

            st.session_state.df_result = df_result
            st.session_state.process_step = 2
            st.rerun()

        elif step == 2:
            # 日付の表示
            st.markdown(f"### 📅 {st.session_state.message}")
    #         st.markdown("### ダウンロード")
    #         st.success("✅ 書類作成完了")
    #         df_result = st.session_state.df_result
    #         template_path = get_template_config()[selected_template][
    #             "template_excel_path"
    #         ]
    #         output_excel = write_values_to_template(
    #             df_result, template_path, st.session_state.extracted_date
    #         )
    #         centered_download_button(
    #             label="📥 Excelファイルをダウンロード",
    #             data=output_excel.getvalue(),
    #             file_name=f"{selected_template_label}_{st.session_state.extracted_date}.xlsx",
    #             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #         )

    # else:
    #     uploaded_count = len(required_keys) - len(missing_keys)
    #     total_count = len(required_keys)

    #     st.progress(uploaded_count / total_count)
    #     st.info(f"📥 {uploaded_count} / {total_count} ファイルがアップロードされました")


from components.custom_button import centered_download_button
from utils.config_loader import get_csv_label_map
from app_pages.manage.view import render_upload_header, show_upload_status
import tempfile


def render_shipping_upload_section():
    csv_label_map = get_csv_label_map()
    shipping_key = "shipping"
    label = csv_label_map.get(shipping_key, "出荷一覧")

    st.markdown("### 📦 出荷一覧ファイルのアップロード")

    render_upload_header(label)
    uploaded_file = st.file_uploader(
        label, type="csv", key=shipping_key, label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            # 一時ファイルに保存（ファイル名を扱えるようにするため）
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # セッションに保存（handle_uploaded_files が使えるように）
            st.session_state[f"uploaded_{shipping_key}"] = tmp_path

        except Exception as e:
            st.error(f"ファイルの保存に失敗しました: {e}")
            st.session_state[f"uploaded_{shipping_key}"] = None
    else:
        st.session_state[f"uploaded_{shipping_key}"] = None


def check_single_file_uploaded(
    uploaded_file: str | None, required_key: str
) -> tuple[bool, str | None]:
    """
    単一ファイルがアップロードされているかをチェックする

    Args:
        uploaded_file (str | None): 一時ファイルパスまたは None
        required_key (str): 対象のファイルキー名（例: 'shipping'）

    Returns:
        is_uploaded (bool): ファイルがアップロードされているか
        missing_key (str | None): 未アップロードの場合はキー名、それ以外は None
    """
    is_uploaded = uploaded_file is not None
    missing_key = None if is_uploaded else required_key
    return is_uploaded, missing_key
