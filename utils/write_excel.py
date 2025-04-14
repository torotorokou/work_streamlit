from openpyxl import load_workbook
from io import BytesIO
import pandas as pd
import numpy as np


def safe_excel_value(value):
    """Excelに書き込める形式に変換するユーティリティ関数"""
    if pd.isna(value) or value is pd.NA or value is np.nan:
        return None
    elif isinstance(value, (dict, list, set)):
        return str(value)
    elif hasattr(value, "strftime"):
        return value.strftime("%Y/%m/%d")
    return value


def write_values_to_template(df: pd.DataFrame, template_path: str) -> BytesIO:
    """
    DataFrameの 'セル' 列を使って、指定セルに '値' を書き込む関数

    Returns:
        BytesIO: 書き込み済みExcelファイルのメモリデータ（Streamlitダウンロード用など）
    """
    wb = load_workbook(template_path)
    ws = wb.active

    for _, row in df.iterrows():
        cell_ref = row["セル"]
        value = safe_excel_value(row["値"])
        cell = ws[cell_ref]

        if isinstance(value, (int, float)) and value == 0:
            cell.value = 0
            cell.number_format = "0"  # 0.00 → 0 に見せる
        else:
            cell.value = value

    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output
