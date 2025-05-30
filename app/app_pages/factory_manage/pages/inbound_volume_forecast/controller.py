import streamlit as st
import pandas as pd
from utils.config_loader import get_path_from_yaml
from utils.get_holydays import get_japanese_holidays
from logic.factory_manage.predict_model_ver2 import (
    train_and_predict_with_holiday,
)
from components.custom_button import centered_button
from logic.manage.utils.upload_handler import handle_uploaded_files
from utils.config_loader import load_factory_required_files
from utils.check_uploaded_csv import (
    render_csv_upload_section,
    check_single_file_uploaded,
)
from logic.factory_manage.make_df import make_sql_db, make_csv
from logic.factory_manage.sql import load_data_from_sqlite


def csv_controller():
    """
    Streamlit上でCSVファイルのアップロードと整合性チェック、
    加工・保存処理を行うコントローラー。
    """
    selected_template = "inbound_volume"

    # --- 必要なファイルキーを設定ファイルから取得 ---
    required_keys = load_factory_required_files()[selected_template]

    # --- アップロードUI表示（テンプレートに応じたファイル形式） ---
    csv_file_type = "receive"
    render_csv_upload_section(csv_file_type)

    # --- ファイルアップロードの整合性チェック ---
    uploaded_files = handle_uploaded_files(required_keys)
    uploaded_file = uploaded_files.get(csv_file_type)
    all_uploaded, missing_key = check_single_file_uploaded(uploaded_file, csv_file_type)
    print(all_uploaded, missing_key)

    if all_uploaded:
        st.success("✅ 必要なファイルがすべてアップロードされました！")
        st.markdown("---")

        if centered_button("⏩ CSVをアップロード"):
            # --- CSVファイル読み込み・整形 ---
            df = pd.read_csv(uploaded_file)
            df = make_csv(df)

            # --- SQLite DBに保存 ---
            make_sql_db(df)

            # --- 完了通知とUIリセット ---
            st.success("📥 CSVファイルの読み込みと保存が完了しました。")
            st.toast("CSV処理が正常に完了しました", icon="📁")

            # --- セッションからアップロード情報を削除し再描画 ---
            key_to_clear = f"uploaded_{csv_file_type}"
            if key_to_clear in st.session_state:
                del st.session_state[key_to_clear]

            st.rerun()


def predict_hannyu_ryou_controller(start_date, end_date):
    """
    搬入量予測のための処理コントローラー。

    - 過去の複数年分のデータを読み込み、統合・整形
    - 指定期間に対する祝日情報の取得
    - 学習・予測ロジックの呼び出し（train_and_predict_with_holiday）

    Parameters:
        start_date (date): 予測対象の開始日
        end_date (date): 予測対象の終了日

    Returns:
        pd.DataFrame: 予測結果のデータフレーム
    """
    # --- データ取得 ---
    # csvから
    df_raw = read_csv_controller()

    # SQLから
    # df_raw = load_data_from_sqlite()

    # --- 祝日データ取得 ---
    holidays = get_japanese_holidays(start=start_date, end=end_date, as_str=True)

    # --- 予測実行 ---
    df_result = train_and_predict_with_holiday(
        df_raw, str(start_date), str(end_date), holidays
    )
    return df_result


def read_csv_controller():
    """
    搬入量予測に必要なCSVデータを読み込んで統合・整形する関数。

    Returns:
        pd.DataFrame: 整形済みの搬入データ（列: 伝票日付・品名・正味重量）
    """

    # --- データ取得 ---
    base_dir = get_path_from_yaml("input", section="directories")

    # --- 新データ ---
    df_raw = pd.read_csv(f"{base_dir}/20240501-20250422.csv", encoding="utf-8")[
        ["伝票日付", "正味重量", "品名"]
    ]

    # --- 旧データ（複数年） ---
    df_2020 = pd.read_csv(f"{base_dir}/2020顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2021 = pd.read_csv(f"{base_dir}/2021顧客.csv")[
        ["伝票日付", "商品", "正味重量_明細"]
    ]
    df_2023 = pd.read_csv(f"{base_dir}/2023_all.csv", low_memory=False)[
        ["伝票日付", "商品", "正味重量_明細"]
    ]

    # --- 統合・整形処理 ---
    df_all = pd.concat([df_2020, df_2021, df_2023])
    df_all["伝票日付"] = pd.to_datetime(df_all["伝票日付"])
    df_all.rename(columns={"商品": "品名", "正味重量_明細": "正味重量"}, inplace=True)
    df_raw = pd.concat([df_raw, df_all])

    # --- 日付・重量クリーニング ---
    df_raw["伝票日付"] = df_raw["伝票日付"].str.replace(r"\(.*\)", "", regex=True)
    df_raw["伝票日付"] = pd.to_datetime(df_raw["伝票日付"], errors="coerce")
    df_raw["正味重量"] = pd.to_numeric(df_raw["正味重量"], errors="coerce")
    df_raw = df_raw.dropna(subset=["正味重量"])

    return df_raw
