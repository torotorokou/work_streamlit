import pandas as pd
from utils.value_setter import set_value_fast


def write_sum_to_target_cell(
    df: pd.DataFrame,
    target_keys: list[str],
    target_values: list,
    value_column: str = "値",
) -> pd.DataFrame:
    """
    指定したキー構成に対応するセルに、合計値を設定する汎用関数。

    Parameters:
        df : pd.DataFrame
            対象データフレーム（通常はマスターCSV）
        target_keys : list[str]
            キー列名のリスト（例: ["業者CD", "業者名", "品名"]）
        target_values : list
            条件として一致させたい値のリスト（例: ["合計", None, None]）
        value_column : str
            合計を算出する対象の値列（デフォルト: "値"）

    Returns:
        pd.DataFrame : 合計が反映されたDataFrame
    """
    total = pd.to_numeric(df[value_column], errors="coerce").sum()

    set_value_fast(df, target_keys, target_values, total, value_col=value_column)

    return df


def summarize_value_by_cell_with_label(
    df: pd.DataFrame,
    value_col: str = "値",
    cell_col: str = "セル",
    label_col: str = "有価名",
) -> pd.DataFrame:
    """
    セル単位で値を集計し、対応するラベル列（例：有価名）を付加した集計結果を返す。

    Parameters
    ----------
    df : pd.DataFrame
        元のデータフレーム（テンプレート含む）
    value_col : str
        数値に変換して合計する列名（例: "値"）
    cell_col : str
        セル位置を示す列名（例: "セル"）
    label_col : str
        ラベル（名前）を示す列名（例: "有価名"）

    Returns
    -------
    pd.DataFrame
        集計された「セル + 値 + ラベル」形式のDataFrame
    """
    # ① 数値変換（混在対応）
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # ② セル単位で合計
    grouped = df.groupby(cell_col, as_index=False)[value_col].sum()

    # ③ セルとラベルの対応表を作成（重複除外）
    cell_to_label = df[[cell_col, label_col]].drop_duplicates()

    # ④ マージ
    grouped_named = pd.merge(grouped, cell_to_label, on=cell_col, how="left")

    return grouped_named
