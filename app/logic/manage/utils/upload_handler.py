import streamlit as st
from logic.detect_csv import detect_csv_type
from components.ui_message import show_warning_bubble
from utils.logger import app_logger  # ロガーのインポート


def handle_uploaded_files(required_keys, csv_label_map):
    """
    アップロードされたCSVファイルの整合性を確認し、正しいもののみを返す関数。

    Parameters:
    - required_keys: アップロードが必要なファイルのキー一覧（例：["receive", "shipping"]）
    - csv_label_map: 各キーに対応するCSV種別名（例：{"receive": "受入データ"}）

    Returns:
    - uploaded_files: キーに対応するアップロード済みファイルの辞書（不正ならNone）
    """
    logger = app_logger()
    uploaded_files = {}

    for key in required_keys:
        uploaded = st.session_state.get(f"uploaded_{key}")
        logger.info(f"checking: {key}, uploaded: {bool(uploaded)}")

        if uploaded:
            expected_name = csv_label_map.get(key, key)
            detected_name = detect_csv_type(uploaded)

            logger.info(f"{key} → expected: {expected_name}, detected: {detected_name}")

            if detected_name != expected_name:
                logger.warning(
                    f"{key} mismatch: expected {expected_name}, got {detected_name}"
                )
                show_warning_bubble(expected_name, detected_name)
                st.session_state[f"uploaded_{key}"] = None
                uploaded_files[key] = None
            else:
                uploaded_files[key] = uploaded
        else:
            logger.warning(f"No file uploaded for: {key}")
            uploaded_files[key] = None

    return uploaded_files
