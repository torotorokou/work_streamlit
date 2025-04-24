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
    label_source_col: str,
    offset: int = -1,
    cell_column: str = "セル",
    value_column: str = "値",
) -> pd.DataFrame:
    """
    任意の列の値をラベルとして使用し、セル位置を指定行だけずらしてラベル行を生成する。

    Parameters:
        df : pd.DataFrame
            元のテンプレートDataFrame
        cell_column : str
            セル位置を記載した列名（例: "セル"）
        label_source_col : str
            ラベルとして "値" に転記する元の列名（例: "業者名", "有価名"）
        offset : int
            セルの行をずらす量（例: -1で上に）
        value_column : str
            ラベルを書き込む対象列（デフォルト: "値"）

    Returns:
        pd.DataFrame
            ラベル行だけを含むDataFrame（他の列はすべて空欄）
    """
    df_label = df.copy()

    # セルをオフセット行だけずらす
    df_label[cell_column] = df_label[cell_column].apply(
        lambda x: shift_cell_row(x, offset)
    )

    # ラベル値を値列にコピー
    df_label[value_column] = df[label_source_col]

    # 値・セル以外の列は空にする
    for col in df_label.columns:
        if col not in [cell_column, value_column]:
            df_label[col] = ""

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
