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
