import json
import os
import yaml
from pathlib import Path


def load_config(config_path="config/config.json") -> dict:
    """
    JSON形式の設定ファイル（config.json）を読み込んで辞書として返す。

    Parameters:
        config_path (str): 設定ファイルへのパス（デフォルト: config/config.json）

    Returns:
        dict: 読み込まれた設定情報

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: JSONの解析に失敗した場合
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")


def load_yaml(filename: str) -> dict:
    """
    指定されたYAMLファイルを辞書形式で読み込む。

    Parameters:
        filename (str): 読み込むYAMLファイル名（config/からの相対パス）

    Returns:
        dict: YAMLから読み込まれた辞書データ
    """
    path = Path("config") / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_app_config() -> dict:
    """アプリケーション設定（app_config.yaml）を読み込む"""
    return load_yaml("app_config.yaml")


def get_path_config() -> dict:
    """パス設定（main_paths.yaml）を読み込む"""
    return load_yaml("main_paths.yaml")


def get_template_config() -> dict:
    """テンプレート設定（templates_config.yaml）を読み込む"""
    return load_yaml("templates_config.yaml")


def get_expected_dtypes() -> dict:
    """各列の期待されるデータ型設定（expected_dtypes.yaml）を読み込む"""
    return load_yaml("expected_dtypes.yaml")


def get_page_config() -> list:
    """
    ページ設定ファイル（page_config.yaml）からページ一覧を取得。

    Returns:
        list[dict]: ページ設定（各要素に label と id を含む）
    """
    return load_yaml("page_config.yaml")["pages"]


def get_page_dicts():
    """
    ページ設定から、以下3つの辞書/リストを生成して返す：
      - page_dict: 表示名 → URL ID
      - page_dict_reverse: URL ID → 表示名
      - page_labels: 表示名のリスト（UI用）

    Returns:
        tuple(dict, dict, list): (page_dict, page_dict_reverse, page_labels)
    """
    pages = get_page_config()
    page_dict = {page["label"]: page["id"] for page in pages}
    page_dict_reverse = {v: k for k, v in page_dict.items()}
    page_labels = list(page_dict.keys())
    return page_dict, page_dict_reverse, page_labels
