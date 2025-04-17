import streamlit as st
from logic.models.csv_processor import process_csv_by_date, check_date_alignment
from components.ui_message import (
    show_success,
    show_warning,
    show_error,
    show_date_mismatch,
)
from utils.file_loader import load_uploaded_csv_files
from utils.cleaners import enforce_dtypes

# from utils.preprocessor import enforce_dtypes
from utils.data_schema import load_expected_dtypes
from utils.config_loader import load_config_json
from utils.logger import app_logger


def prepare_csv_data(uploaded_files: dict, date_columns: dict) -> dict:
    """
    アップロードされた CSV ファイル群を読み込み、各データフレームの型を整形し、
    日付処理および日付整合性チェックを行います。
    処理途中で一旦 success メッセージを表示し、最終的にのみ最終結果の success メッセージを残します。

    Parameters:
        uploaded_files (dict): アップロードされたファイル群
        date_columns (dict): 各ファイルの対応する日付カラム名

    Returns:
        dict: 前処理後のデータフレーム辞書（問題があれば空辞書）
    """
    # ステータス用のプレースホルダーを用意（成功メッセージ用）
    logger = app_logger()

    # --- 書類作成の開始メッセージ ---
    logger.info("📄 これからCSVの書類を作成します...")
    dfs = load_uploaded_csv_files(uploaded_files)

    config = load_config_json()
    expected_dtypes = load_expected_dtypes(config)

    for key in dfs:
        dfs[key] = enforce_dtypes(dfs[key], expected_dtypes)

    # --- CSVの日付確認中 ---
    logger.info("📄 CSVの日付を確認中です...")
    for key, df in dfs.items():
        date_col = date_columns.get(key)

        if not date_col:
            show_warning(f"⚠️ {key} の日付カラム定義が存在しません。")
            return {}

        if date_col not in df.columns:
            show_warning(f"⚠️ {key} のCSVに「{date_col}」列が見つかりませんでした。")
            return {}

        dfs[key] = process_csv_by_date(df, date_col)

    result = check_date_alignment(dfs, date_columns)
    if not result["status"]:
        show_error(result["error"])
        if "details" in result:
            show_date_mismatch(result["details"])
        return {}

    # --- 中間の成功メッセージをクリアし、最終メッセージのみ表示 ---
    logger.info(f"✅ すべてのCSVで日付が一致しています：{result['dates']}")
    return dfs
