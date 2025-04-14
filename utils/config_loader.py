import json
import os


def load_config(config_path="config/config.json") -> dict:
    """
    config.json を読み込んで辞書として返す関数。

    Parameters:
        config_path (str): 設定ファイルのパス（相対パスで指定）

    Returns:
        dict: 読み込まれた設定内容
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")
