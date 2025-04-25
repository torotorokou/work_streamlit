import pandas as pd
from utils.logger import app_logger


def summary_merge_by_keys(
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


def summary_apply_by_sheet(
    master_csv: pd.DataFrame,
    data_df: pd.DataFrame,
    sheet_name: str,
    key_cols: list[str],
    source_col: str = "正味重量",
    target_col: str = "値",
) -> pd.DataFrame:
    """
    マスターCSVの特定シート部分に対して、外部データをgroupby集計してマージし、値を書き込む汎用関数。

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
    merged_df = summary_merge_by_keys(
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
