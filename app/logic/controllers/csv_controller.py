import pandas as pd
from logic.models.csv_processor import process_csv_by_date, check_date_alignment
from components.ui_message import (
    show_warning,
    show_error,
    show_date_mismatch,
)
from utils.file_loader import load_uploaded_csv_files
from utils.cleaners import enforce_dtypes, strip_whitespace
from utils.config_loader import get_expected_dtypes_by_template
from utils.logger import app_logger


def apply_expected_dtypes(
    dfs: dict[str, pd.DataFrame],
    template_key: str,
) -> dict[str, pd.DataFrame]:
    """
    アップロードされた各CSVに対して、テンプレート定義に基づきデータ型を強制適用する。

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        読み込んだCSVファイル群
    template_key : str
        テンプレート名（例: average_sheet）

    Returns
    -------
    dict[str, pd.DataFrame]
        型強制後のDataFrame群
    """
    logger = app_logger()
    expected_dtypes_per_file = get_expected_dtypes_by_template(template_key)

    for key in dfs:
        dfs[key] = strip_whitespace(dfs[key])  # 🔽 空白除去

        dtypes = expected_dtypes_per_file.get(key)
        if dtypes:
            dfs[key] = enforce_dtypes(dfs[key], dtypes)
            logger.info(f"✅ 型を適用しました: {key}")

    return dfs


def prepare_csv_data(
    uploaded_files: dict, date_columns: dict, template_key: str
) -> dict:
    logger = app_logger()
    logger.info("📄 これからCSVの書類を作成します...")

    dfs = load_uploaded_csv_files(uploaded_files)

    # 型適用処理を独立関数で実施
    dfs = apply_expected_dtypes(dfs, template_key)

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

    logger.info(f"✅ すべてのCSVで日付が一致しています：{result['dates']}")
    return dfs, result["dates"]
