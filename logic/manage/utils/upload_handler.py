import streamlit as st
from logic.detect_csv import detect_csv_type
from components.ui_message import show_warning_bubble


def handle_uploaded_files(required_keys, csv_label_map, header_csv_path):
    """
    アップロードされたCSVファイルの整合性を確認し、正しいもののみを返す関数。

    Parameters:
    - required_keys: アップロードが必要なファイルのキー一覧（例：["receive", "shipping"]）
    - csv_label_map: 各キーに対応するCSV種別名（例：{"receive": "受入データ"}）
    - header_csv_path: CSV種別を判別するための基準ヘッダー定義ファイルのパス

    Returns:
    - uploaded_files: キーに対応するアップロード済みファイルの辞書（不正ならNone）
    """
    
    uploaded_files = {}  # 各キーごとに検証済みのファイルを格納する辞書

    for key in required_keys:
        # セッションステートから該当ファイルを取得（例: uploaded_receive）
        uploaded = st.session_state.get(f"uploaded_{key}")

        if uploaded:
            # 期待されるCSVの種別名（なければキー名を使う）
            expected_name = csv_label_map.get(key, key)

            # アップロードされたファイルのCSV種別を判定
            detected_name = detect_csv_type(uploaded, header_csv_path)
        
            if detected_name != expected_name:
                # 判定結果が一致しない場合は警告を表示し、ファイルを無効化
                show_warning_bubble(expected_name, detected_name)
                st.session_state[f"uploaded_{key}"] = None  # アップロード欄をリセット
                uploaded_files[key] = None  # このキーのファイルは不正とする
            else:
                # 判定が一致すれば、有効なファイルとして記録
                uploaded_files[key] = uploaded
        else:
            # ファイルがアップロードされていない場合はNoneを設定
            uploaded_files[key] = None

    # 有効なファイルだけを含む辞書を返す（不正または未アップロードはNone）
    return uploaded_files
