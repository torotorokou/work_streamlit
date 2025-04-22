from openpyxl import load_workbook
from io import BytesIO
import pandas as pd
import numpy as np
from datetime import datetime
from openpyxl.cell.cell import MergedCell
from utils.logger import app_logger  # ロガーを使っていれば


def safe_excel_value(value):
    """Excelに書き込める形式に変換するユーティリティ関数"""
    if pd.isna(value) or value is pd.NA or value is np.nan:
        return None
    elif isinstance(value, (dict, list, set)):
        return str(value)
    elif hasattr(value, "strftime"):
        return value.strftime("%Y/%m/%d")
    return value


def write_values_to_template(
    df: pd.DataFrame, template_path: str, extracted_date
) -> BytesIO:
    """
    DataFrameの 'セル' 列を使って、指定セルに '値' を書き込む関数。
    さらに、シート名を「YYYYMMDD」に変更する。

    Returns:
        BytesIO: 書き込み済みExcelファイルのメモリデータ（Streamlitダウンロード用など）
    """
    logger = app_logger()
    wb = load_workbook(template_path)
    ws = wb.active

    # --- セルへの書き込み ---
    for idx, row in df.iterrows():
        cell_ref = row["セル"]
        value = safe_excel_value(row["値"])
        try:
            cell = ws[cell_ref]

            if isinstance(cell, MergedCell):
                logger.warning(
                    f"セル {cell_ref} は結合セル（MergedCell）で書き込み不可。スキップしました。値: {value}"
                )
                continue

            if isinstance(value, (int, float)) and value == 0:
                cell.value = 0
                cell.number_format = "0"
            else:
                cell.value = value

        except Exception as e:
            logger.error(
                f"セル {cell_ref} への書き込みでエラーが発生: 値={value} / エラー={e}"
            )

    # --- シート名を今日の日付に変更（例：20250414） ---
    ws.title = extracted_date

    # --- メモリ上に保存 ---
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output
