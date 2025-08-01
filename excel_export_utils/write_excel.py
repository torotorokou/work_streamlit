"""
Excel出力ユーティリティモジュール

このモジュールは、DataFrameをExcelテンプレートに書き込むための機能を提供します。
主な機能：
- Excelテンプレートの読み込み
- DataFrameデータをExcelセルに書き込み
- セルの書式保持
- BytesIOとしての出力
"""

from openpyxl import load_workbook
from io import BytesIO
import pandas as pd
import numpy as np
from openpyxl.cell.cell import MergedCell
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from copy import copy
from pathlib import Path
import logging
from typing import Optional


def setup_logger():
    """簡易ロガーのセットアップ"""
    logger = logging.getLogger("excel_export")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def safe_excel_value(value):
    """Excelに書き込める形式に変換するユーティリティ関数"""
    if pd.isna(value) or value is pd.NA or value is np.nan:
        return None
    elif isinstance(value, (dict, list, set)):
        return str(value)
    elif hasattr(value, "strftime"):
        return value.strftime("%Y/%m/%d")
    return value


def load_template_workbook(template_path: str | Path) -> Workbook:
    """Excelテンプレートファイルを読み込み"""
    if isinstance(template_path, str):
        template_path = Path(template_path)

    if not template_path.exists():
        raise FileNotFoundError(
            f"テンプレートファイルが見つかりません: {template_path}"
        )

    return load_workbook(template_path)


def write_dataframe_to_worksheet(df: pd.DataFrame, ws: Worksheet, logger=None):
    """
    DataFrameのデータをワークシートに書き込み

    Args:
        df: 書き込むDataFrame（"セル"列と"値"列が必要）
        ws: 対象のワークシート
        logger: ロガー（オプション）
    """
    if logger is None:
        logger = setup_logger()

    for idx, row in df.iterrows():
        cell_ref = row.get("セル")
        value = safe_excel_value(row.get("値"))

        if pd.isna(cell_ref) or str(cell_ref).strip() in ["", "未設定"]:
            logger.info(f"空欄または未設定のセルはスキップされました。行 {idx}")
            continue

        try:
            # セル参照を正しく解析
            if ":" in str(cell_ref):
                # 範囲指定の場合はスキップ
                logger.warning(f"範囲指定セル {cell_ref} はスキップしました")
                continue

            cell = ws[str(cell_ref)]

            # セル配列の場合、最初のセルを取得
            if isinstance(cell, tuple):
                cell = cell[0] if len(cell) > 0 else None

            if cell is None:
                logger.warning(f"セル {cell_ref} が取得できませんでした")
                continue

            if isinstance(cell, MergedCell):
                logger.warning(f"セル {cell_ref} は結合セルで書き込み不可。値: {value}")
                continue

            # --- 書式をdeep copyで保持 ---
            try:
                original_font = copy(cell.font)
                original_fill = copy(cell.fill)
                original_border = copy(cell.border)
                original_format = cell.number_format

                # 値の上書き
                cell.value = value

                # --- 書式の復元 ---
                cell.font = original_font
                cell.fill = original_fill
                cell.border = original_border
                cell.number_format = original_format
            except Exception as format_error:
                # 書式保持に失敗した場合は値のみ設定
                logger.warning(
                    f"セル {cell_ref} の書式保持に失敗、値のみ設定: {format_error}"
                )
                cell.value = value

        except Exception as e:
            logger.error(f"セル {cell_ref} 書き込み失敗: {e} / 値: {value}")


def rename_sheet(wb: Workbook, new_title: str):
    """アクティブシートの名前を変更"""
    ws = wb.active
    if ws is not None:
        ws.title = new_title


def save_workbook_to_bytesio(wb: Workbook) -> BytesIO:
    """ワークブックをBytesIOに保存"""
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output


def write_values_to_template(
    df: pd.DataFrame, template_path: str, extracted_date: str | None = None
) -> BytesIO:
    """
    単一責任原則に基づいて分割されたExcelテンプレート書き込み関数

    Args:
        df: 書き込むDataFrame（"セル"列と"値"列が必要）
        template_path: Excelテンプレートファイルのパス
        extracted_date: シート名に使用する日付（オプション）

    Returns:
        BytesIO: Excelファイルのバイナリデータ
    """
    logger = setup_logger()
    logger.info(f"Excelテンプレート処理開始: {template_path}")

    # テンプレート読み込み
    wb = load_template_workbook(template_path)
    ws = wb.active

    if ws is None:
        raise ValueError("テンプレートファイルにアクティブシートが見つかりません")

    # セルへの書き込み
    write_dataframe_to_worksheet(df, ws, logger=logger)

    # シート名変更（日付が指定されている場合）
    if extracted_date:
        rename_sheet(wb, extracted_date)

    # メモリ出力
    result = save_workbook_to_bytesio(wb)
    logger.info("Excelテンプレート処理完了")

    return result
