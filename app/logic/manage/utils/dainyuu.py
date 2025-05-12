import pandas as pd
from utils.value_setter import set_value_fast, set_value_fast_safe


def get_search_word(df: pd.DataFrame, key_col: str, key_value: str, search_col: str) -> str:
    """
    指定されたキー列と値に一致する行から、検索ワード列の値を取得する。

    Parameters:
        df (pd.DataFrame): 検索対象のデータフレーム
        key_col (str): 行を絞り込むためのキー列名（例: "大項目"）
        key_value (str): 絞り込む対象の値（例: "有価物"）
        search_col (str): 取得したい列名（例: "検索ワード1"）

    Returns:
        str: 該当する検索ワードの値（見つからなければ None）
    """
    # 該当する行の search_col 列を取得
    filtered = df.loc[df[key_col] == key_value, search_col]
    return filtered.values[0] if not filtered.empty else None


def get_value_by_key(df: pd.DataFrame, key_col: str, key_value: str, value_col: str) -> float:
    """
    指定されたキー列と値に一致する行から、対象列の値を取得する。

    Parameters:
        df (pd.DataFrame): 実績データ等のデータフレーム
        key_col (str): 行を絞り込むためのキー列名（例: "大項目"）
        key_value (str): 絞り込む対象の値（例: "スクラップ"）
        value_col (str): 取得したい数値列名（例: "値"）

    Returns:
        float: 該当する値（見つからなければ None）
    """
    # 該当する行の value_col 列を取得
    filtered = df.loc[df[key_col] == key_value, value_col]
    return filtered.values[0] if not filtered.empty else None


def apply_value_to_master(
    df: pd.DataFrame,
    key_cols: list[str],
    key_values: list[str],
    value: float
) -> pd.DataFrame:
    """
    指定されたキー列と値に一致する行の対象セルを更新する。

    Parameters:
        df (pd.DataFrame): 更新対象のデータフレーム（例: master_csv）
        key_cols (list[str]): 絞り込みに使う列名のリスト（例: ["大項目"]）
        key_values (list[str]): 各 key_col に対応する値のリスト（例: ["有価物"]）
        value (float): 更新する値

    Returns:
        pd.DataFrame: 更新後のデータフレーム
    """
    # set_value_fast_safe に委譲して対象セルの値を更新
    return set_value_fast_safe(df, key_cols, key_values, value)



def apply_summary_by_item(
    master_csv: pd.DataFrame,
    csv_fac: pd.DataFrame,
    item_name: str,
    key_col: str = "大項目",
    search_col: str = "検索ワード1",
    value_col: str = "値",
) -> pd.DataFrame:
    """
    指定された項目名に対応する検索ワードと値を取得し、master_csvに反映。

    Parameters:
        master_csv: テンプレート元CSV
        csv_fac: 実績データ
        item_name: 更新対象の項目名（例：'有価物'）
        key_col: 検索キー列（デフォルト: '大項目'）
        search_col: master_csv側の検索ワード列（デフォルト: '検索ワード1'）
        value_col: csv_fac側の値列（デフォルト: '値'）

    Returns:
        更新後の master_csv
    """
    search_word = get_search_word(master_csv, key_col, item_name, search_col)
    value = get_value_by_key(csv_fac, key_col, search_word, value_col)
    return apply_value_to_master(master_csv, [key_col], [item_name], value)
