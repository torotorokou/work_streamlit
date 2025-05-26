import streamlit as st

# ✅ 標準ライブラリ

# ✅ サードパーティ
import pandas as pd

# ✅ プロジェクト内 - components（UI共通パーツ）
from components.custom_button import centered_button
from components.custom_progress_bar import CustomProgressBar

# ✅ プロジェクト内 - view（UIビュー）

# ✅ プロジェクト内 - logic（処理・データ変換など）
from logic.manage.utils.upload_handler import handle_uploaded_files

# ✅ プロジェクト内 - utils（共通ユーティリティ）
from utils.debug_tools import save_debug_parquets
from utils.config_loader import (
    get_csv_label_map,
)

from utils.config_loader import load_factory_required_files
from app_pages.factory_manage.pages.balance_management_table.process import (
    processor_func,
)
from components.custom_button import centered_download_button
from io import BytesIO


def factory_manage_controller():
    file_name = "工場収支モニタリング表"
    st.subheader(f"🗑 {file_name}")
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
            message = f"{target_year}年{target_month}月：現在数量"
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
            # st.markdown(f"### 📅 {st.session_state.message}")

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
            st.markdown("### ダウンロード")
            st.success("✅ 書類作成完了")

            # ✅ セッションからdf_resultを取得
            df_result = st.session_state.get("df_result")

            if df_result is None:
                st.error(
                    "❌ 書類データが存在しません。前のステップで処理が完了しているか確認してください。"
                )
                return

            excel_bytes = convert_df_to_excel_bytes(df_result)

            centered_download_button(
                label="📥 Excelファイルをダウンロード",
                data=excel_bytes,
                file_name=f"{file_name}_{st.session_state.extracted_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


from app_pages.manage.view import render_upload_header
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


from io import BytesIO
import pandas as pd


def convert_df_to_excel_bytes(df: pd.DataFrame) -> BytesIO:
    """
    DataFrameをExcel形式のBytesIOに変換

    - 中項目のNaNは空白に
    - 游ゴシックフォント
    - 単価は小数点2桁表示
    - 全列同じ幅に揃える
    - 罫線なし
    """
    output = BytesIO()

    # --- NaNや文字列'nan'などを空白に変換（中項目のみ）
    if "中項目" in df.columns:
        df = df.copy()
        df["中項目"] = (
            df["中項目"]
            .replace(["nan", "NaN", "None"], "")  # ← 文字列としてのnanも空白に
            .fillna("")  # ← 本物のNaNも空白に
            .astype(str)  # ← 念のためすべて文字列化
        )

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1", startrow=1, header=False)

        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]

        # --- フォント定義（游ゴシック、罫線なし）
        header_format = workbook.add_format(
            {"font_name": "游ゴシック", "bold": True, "bg_color": "#F2F2F2"}
        )

        cell_format = workbook.add_format({"font_name": "游ゴシック"})

        unit_price_format = workbook.add_format(
            {"font_name": "游ゴシック", "num_format": "#,##0.00"}
        )

        # --- ヘッダー書き込み
        for col_num, column_name in enumerate(df.columns):
            worksheet.write(0, col_num, column_name, header_format)

        # --- データ書き込み（単価だけフォーマットを分ける）
        for row_num in range(len(df)):
            for col_num in range(len(df.columns)):
                col_name = df.columns[col_num]
                value = df.iat[row_num, col_num]

                if col_name == "単価":
                    worksheet.write(row_num + 1, col_num, value, unit_price_format)
                else:
                    worksheet.write(row_num + 1, col_num, value, cell_format)

        # --- 列幅を個別に指定（列名 → 幅）
        column_widths = {
            "大項目": 15,
            "中項目": 10,
            "合計正味重量": 10,
            "合計金額": 10,
            "単価": 7,
            "台数": 7,
        }

        for i, col_name in enumerate(df.columns):
            width = column_widths.get(col_name, 20)  # 未定義なら幅20に
            worksheet.set_column(i, i, width)

    output.seek(0)
    return output
