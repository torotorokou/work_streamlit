from openpyxl import load_workbook
from io import BytesIO
import pandas as pd


def write_values_to_template(df: pd.DataFrame, template_path: str) -> BytesIO:
    """
    DataFrameの 'セル' 列を使って、指定セルに '値' を書き込む関数

    Parameters:
    ----------
    df : pd.DataFrame
        'セル' 列にセル番地（例：A2）、'値' 列に書き込み内容があるDataFrame
    template_path : str
        ベースとなるExcelテンプレートのファイルパス

    Returns:
    -------
    BytesIO
        書き込み済みExcelファイルのメモリデータ（Streamlitダウンロードなどに使用可）
    """
    wb = load_workbook(template_path)
    ws = wb.active

    for _, row in df.iterrows():
        cell_ref = row["セル"]
        value = row["値"]
        ws[cell_ref] = value

    # メモリ上に保存
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output
