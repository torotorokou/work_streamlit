import pandas as pd
from utils.value_setter import set_value_fast
from utils.logger import app_logger


def write_sum_to_target_cell(
    df: pd.DataFrame,
    target_keys: list[str],
    target_values: list,
    value_column: str = "値",
) -> pd.DataFrame:
    """
    テンプレートの「合計」セルに値を書き込みたい。
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
    表の中の数値を、セルごとに合計して、ラベル（名前）も一緒に表示する関数。

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

def summary_apply_by_sheet(
    master_csv: pd.DataFrame,
    data_df: pd.DataFrame,
    sheet_name: str,
    key_cols: list[str],
    source_col: str = "正味重量",
    target_col: str = "値",
) -> pd.DataFrame:
    """
    インポートCSVをgroupby＆sumし、マスターCSVの特定シートの値に書き込む汎用関数。

    Parameters
    ----------
    master_csv : pd.DataFrame
        全体のテンプレートCSV（複数シートを含む）
    data_df : pd.DataFrame
        処理対象のデータ（例：df_shipping）
    sheet_name : str
        処理対象とする "CSVシート名"（例："出荷"）
    key_cols : list[str]
        groupbyキー ＝ マージキー（例：["品名"], ["業者名", "品名"]）
    source_col : str
        集計対象の列（例："正味重量"）
    target_col : str
        書き込み先の列（例："値"）

    Returns
    -------
    pd.DataFrame
        処理済みのマスターCSV（"CSVシート名"以外も含む）
    """
    logger = app_logger()
    logger.info(
        f"▶️ 処理対象シート: {sheet_name}, キー: {key_cols}, 集計列: {source_col}"
    )

    # ① 該当シート部分を取り出す
    target_df = master_csv[master_csv["CSVシート名"] == sheet_name].copy()

    # ② groupbyで合計
    agg_df = data_df.groupby(key_cols, as_index=False)[[source_col]].sum()

    # ③ 安全にマージ
    merged_df = summary_tools_by_keys(
        master_df=target_df, data_df=agg_df, key_cols=key_cols
    )

    # ④ 値を書き込み（NaN以外）
    merged_df = summary_update_column_if_notna(merged_df, source_col, target_col)

    # ⑤ 不要列を削除
    merged_df.drop(columns=[source_col], inplace=True)

    # ⑥ 元に戻す：シート以外はそのまま
    master_others = master_csv[master_csv["CSVシート名"] != sheet_name]
    final_df = pd.concat([master_others, merged_df], ignore_index=True)

    return final_df


def summary_tools_by_keys(
    master_df: pd.DataFrame,
    data_df: pd.DataFrame,
    key_cols: list[str],
    how: str = "left",
) -> pd.DataFrame:
    """
    指定した複数のキー列を使って、安全にマージする関数。
    マスター側のキー列に欠損値（NaN）がある行はマージ対象から除外される。

    Parameters
    ----------
    master_df : pd.DataFrame
        マージ元（テンプレート）
    data_df : pd.DataFrame
        マージ対象のデータ
    key_cols : list[str]
        結合に使用するキー列（1〜3列想定）
    how : str
        マージ方法（デフォルト: "left"）

    Returns
    -------
    pd.DataFrame
        マージ済みのDataFrame（未マージ行も含まれる）
    """

    # ① キーに空欄がある行を除外してマージ
    master_valid = master_df.dropna(subset=key_cols)
    data_valid = data_df.dropna(subset=key_cols)

    merged = pd.merge(master_valid, data_valid, on=key_cols, how=how)

    # ② キーが不完全（NaN含む）な行を保持して復元
    master_skipped = master_df[master_df[key_cols].isna().any(axis=1)]

    # ③ マージしたものと未マージのものを結合して返す
    final_df = pd.concat([merged, master_skipped], ignore_index=True)

    return final_df


def summary_update_column_if_notna(
    df, source_col: str, target_col: str
) -> pd.DataFrame:
    mask = df[source_col].notna()
    df.loc[mask, target_col] = df.loc[mask, source_col]
    return df
