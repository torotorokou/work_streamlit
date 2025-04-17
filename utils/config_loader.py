import json
import os
import yaml
from pathlib import Path


# def load_config_json(config_path="config/config.json") -> dict:
#     """
#     JSON形式の設定ファイル（config.json）を読み込んで辞書として返す。

#     Parameters:
#         config_path (str): 設定ファイルへのパス（デフォルト: config/config.json）

#     Returns:
#         dict: 読み込まれた設定情報

#     Raises:
#         FileNotFoundError: ファイルが存在しない場合
#         ValueError: JSONの解析に失敗した場合
#     """
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

#     with open(config_path, encoding="utf-8") as f:
#         try:
#             return json.load(f)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")


def load_yaml(filename: str) -> dict:
    """
    指定されたYAMLファイルを辞書形式で読み込む。

    Parameters:
        filename (str): 読み込むYAMLファイル名（config/からの相対パス）

    Returns:
        dict: YAMLから読み込まれた辞書データ
    """
    path = Path(filename)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path_config() -> dict:
    """main_paths.yamlを辞書として取得"""
    return load_yaml("config/main_paths.yaml")


def get_app_config() -> dict:
    """main_paths.yaml 経由で app_config を読み込む"""
    config_path = get_path_config()["config_files"]["app_config"]
    return load_yaml(config_path)


def get_expected_dtypes() -> dict:
    """main_paths.yaml 経由で expected_dtypes.yaml を読み込む"""
    config_path = get_path_config()["config_files"]["expected_dtypes"]
    return load_yaml(config_path)


def get_template_config() -> dict:
    """main_paths.yaml 経由で templates_config.yaml を読み込む"""
    config_path = get_path_config()["config_files"]["templates_config"]
    return load_yaml(config_path)


def get_page_config() -> list:
    """main_paths.yaml 経由で page_config.yaml のページ定義を取得"""
    config_path = get_path_config()["config_files"]["page_config"]
    return load_yaml(config_path)["pages"]


def get_page_dicts():
    """
    ページ設定（page_config.yaml）から
    - 表示名→IDの辞書
    - ID→表示名の逆引き
    - 表示名リスト（UI用）
    を返す
    """
    pages = get_page_config()
    page_dict = {p["label"]: p["id"] for p in pages}
    reverse_dict = {v: k for k, v in page_dict.items()}
    labels = list(page_dict.keys())
    return page_dict, reverse_dict, labels


def get_csv_sources_config() -> dict:
    config_path = get_path_config()["config_files"]["csv_sources_config"]
    return load_yaml(config_path)


def get_csv_label_map() -> dict:
    config = get_csv_sources_config()
    return {key: value["label"] for key, value in config.items()}


def get_csv_date_columns() -> dict:
    """
    各CSVファイル種別に対応する日付カラム名を取得する。

    Returns:
        dict: { "receive": "伝票日付", "yard": "伝票日付", ... } の形式
    """
    config = get_csv_sources_config()
    return {key: value["date_column"] for key, value in config.items()}


def get_required_files_map() -> dict:
    """
    各テンプレートに必要なファイル（required_files）を辞書形式で取得。

    Returns:
        dict: テンプレートキー → 必須ファイルリスト
    """
    config = get_template_config()
    return {key: value.get("required_files", []) for key, value in config.items()}


def get_template_descriptions() -> dict:
    config = get_template_config()
    return {key: value.get("description", []) for key, value in config.items()}


def get_template_dict() -> dict:
    """
    テンプレートの表示ラベル → テンプレートキー の辞書を返す。

    Returns:
        dict: 例 {"工場日報": "factory_report", ...}
    """
    config = get_template_config()
    return {value["label"]: key for key, value in config.items()}


def get_expected_dtypes_by_template(template_key: str) -> dict:
    config = get_expected_dtypes()
    return config.get(template_key, {})
