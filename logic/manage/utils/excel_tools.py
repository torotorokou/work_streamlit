import re
import pandas as pd


def shift_cell_row(cell: str, offset: int) -> str:
    match = re.match(r"([A-Z]+)(\d+)", cell)
    if match:
        col, row = match.groups()
        new_row = max(1, int(row) + offset)
        return f"{col}{new_row}"
    return cell


def create_label_rows_generic(
    df: pd.DataFrame,
    key_columns: list[str],
    cell_column: str = "セル",
    label_source_col: str = None,
    offset: int = -1,
    value_column: str = "値"
) -> pd.DataFrame:
    """
    任意のキー列構造に対応したラベル行を作成し、セル位置をずらす。

    Parameters:
        df (pd.DataFrame): 元のテンプレートDataFrame
        key_columns (list[str]): ["大項目", "小項目1", "小項目2"] など、テンプレートのキー列
        cell_column (str): セル位置を記載した列名（例: "セル"）
        label_source_col (str): "値" に転記する元の列名（例: "小項目1"）※Noneなら key_columns[1] を使う
        offset (int): セルの行をずらす量（例: -1で上に）
        value_column (str): 値を書き込む対象列（デフォルト: "値"）

    Returns:
        pd.DataFrame: ラベル行だけを含むDataFrame
    """
    df_label = df.copy()
    df_label[cell_column] = df_label[cell_column].apply(lambda x: shift_cell_row(x, offset))

    if label_source_col is None:
        label_source_col = key_columns[1]  # 通常は "小項目1"

    df_label[value_column] = df[label_source_col]

    # キー列の一部を初期化
    for col in key_columns[1:]:
        df_label[col] = ""
    df_label[key_columns[0]] = None  # "大項目" に相当

    return df_label



def sort_by_cell_row(df: pd.DataFrame, cell_col: str = "セル") -> pd.DataFrame:
    """
    セル番地列の「行番号（数字部分）」をもとに DataFrame をソートする。

    Parameters
    ----------
    df : pd.DataFrame
        並び替え対象のデータフレーム
    cell_col : str
        セル番地が格納されている列名（デフォルトは "セル"）

    Returns
    -------
    pd.DataFrame
        行番号で昇順にソートされたDataFrame（インデックスはリセットされる）
    """
    df = df.copy()
    df["_セル行"] = df[cell_col].apply(lambda x: int(re.findall(r"\d+", x)[0]))
    df = df.sort_values("_セル行").drop(columns="_セル行").reset_index(drop=True)
    return df
