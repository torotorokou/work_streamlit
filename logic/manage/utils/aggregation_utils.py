import pandas as pd
from utils.logger import app_logger


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
