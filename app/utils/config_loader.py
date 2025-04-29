import yaml
import pandas as pd
from pathlib import Path
from utils.type_converter import resolve_dtype
import os
# from utils.logger import app_logger


def load_yaml(filename: str) -> dict:
    """
    指定されたYAMLファイルを辞書形式で読み込む。

    Parameters:
        filename (str): 読み込むYAMLファイル名（config/からの相対パス）

    Returns:
        dict: YAMLから読み込まれた辞書データ
    """
    base_dir = os.getenv("BASE_DIR", "/work/app")  # ← デフォルトもつけると安心
    path = Path(base_dir) / filename
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
    """main_paths.yaml 経由で expected_dtypes.yaml を読み込み、型を解決する"""
    config_path = get_path_config()["config_files"]["expected_dtypes"]
    raw_yaml = load_yaml(config_path)

    resolved = {}
    for template_key, file_map in raw_yaml.items():
        resolved[template_key] = {}
        for file_key, dtype_map in file_map.items():
            resolved[template_key][file_key] = {
                col: resolve_dtype(dtype_str) for col, dtype_str in dtype_map.items()
            }

    return resolved


def get_template_config() -> dict:
    """main_paths.yaml 経由で templates_config.yaml を読み込む"""
    config_path = get_path_config()["config_files"]["templates_config"]
    return load_yaml(config_path)


def get_page_config() -> list:
    """main_paths.yaml 経由で page_config.yaml のページ定義を取得"""
    config_path = get_path_config()["config_files"]["page_config"]
    return load_yaml(config_path)["pages"]


def get_unit_price_table_csv() -> pd.DataFrame:
    """
    単価表CSVを読み込んでDataFrameとして返す。
    """
    csv_path = get_path_config()["csv"]["unit_price_table"]
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df


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
    """
    指定されたテンプレートキーに対応するCSVファイルごとの
    カラム型定義（expected_dtypes）を返す。

    この関数は expected_dtypes.yaml を読み込んだ結果から、
    指定テンプレートに対応する型定義だけを抽出して返します。

    Parameters:
        template_key (str): テンプレート名（例: "average_sheet", "factory_report"）

    Returns:
        dict: ファイルキーごとのカラム名と型の辞書。
              例:
              {
                  "receive": {
                      "金額": float,
                      "正味重量": int,
                      "伝票日付": "datetime64[ns]"
                  },
                  "yard": {
                      "品名": str,
                      ...
                  }
              }
              対応テンプレートが存在しない場合は空の辞書を返します。
    """
    config = get_expected_dtypes()
    return config.get(template_key, {})


def get_required_columns_definition(template_name: str) -> dict:
    config_path = get_path_config()["config_files"]["required_columns_definition"]
    all_defs = load_yaml(config_path)
    return all_defs.get(template_name, {})
