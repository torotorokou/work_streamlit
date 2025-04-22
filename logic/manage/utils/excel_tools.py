import re
import pandas as pd


def shift_cell_row(cell: str, offset: int) -> str:
    match = re.match(r"([A-Z]+)(\d+)", cell)
    if match:
        col, row = match.groups()
        new_row = max(1, int(row) + offset)
        return f"{col}{new_row}"
    return cell


def create_label_rows(df: pd.DataFrame, offset: int = -1) -> pd.DataFrame:
    """
    マスターCSVからラベル行（小項目1の内容をセルに記入）を作成し、セル位置を行方向にずらす。

    Parameters
    ----------
    df : pd.DataFrame
        元のマスターCSV
    offset : int
        セルの行をずらす量（-1 で上に、+1 で下に追加）

    Returns
    -------
    pd.DataFrame
        生成されたラベル行だけのDataFrame
    """

    df_label = df.copy()
    df_label["セル"] = df_label["セル"].apply(lambda x: shift_cell_row(x, offset))
    df_label["値"] = df["小項目1"]
    df_label[["小項目1", "小項目2", "小項目3"]] = ""
    df_label["大項目"] = None

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
