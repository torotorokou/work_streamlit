import pandas as pd


def make_df_shipping_after_use(
    master_csv: pd.DataFrame, df_shipping: pd.DataFrame
) -> pd.DataFrame:
    """出荷データをマスターデータに基づいてフィルタリングし、必要な情報を追加する

    Args:
        master_csv (pd.DataFrame): マスターデータ（業者CD、品名、運搬社数などを含む）
        df_shipping (pd.DataFrame): 出荷データ

    Returns:
        pd.DataFrame: フィルタリングと情報追加が完了した出荷データ
    """

    def _filter_by_vendor_code(df: pd.DataFrame) -> pd.DataFrame:
        """業者CDでフィルタリング"""
        return df[df["業者CD"].isin(master_csv["業者CD"])].copy()

    def _filter_by_item_name(df: pd.DataFrame) -> pd.DataFrame:
        """品名指定がある場合のフィルタリング"""
        # 品名指定があるものを抽出
        item_filter_df = master_csv[master_csv["品名"].notna()][
            ["業者CD", "品名"]
        ].drop_duplicates()

        if item_filter_df.empty:
            return df

        # 業者CDと品名のペアが一致する行のみ残す
        df = df.merge(item_filter_df, on=["業者CD", "品名"], how="left", indicator=True)
        df = df[
            (df["_merge"] == "both") | (~df["業者CD"].isin(item_filter_df["業者CD"]))
        ]
        return df.drop(columns=["_merge"])

    def _add_transport_info(df: pd.DataFrame) -> pd.DataFrame:
        """運搬関連情報の追加"""
        # 運搬社数を追加（業者CDごとに1件に絞る）
        unique_master = master_csv.drop_duplicates(subset=["業者CD"])[
            ["業者CD", "運搬社数"]
        ]
        df = df.merge(unique_master, on="業者CD", how="left")

        # 運搬費カラムを初期化
        df["運搬費"] = 0
        return df

    # メイン処理の実行
    df_after = _filter_by_vendor_code(df_shipping)
    df_after = _filter_by_item_name(df_after)

    # 正味重量が0のデータを除外
    df_after = df_after[df_after["正味重量"].fillna(0) != 0]

    # 運搬情報の追加
    df_after = _add_transport_info(df_after)

    # 業者CDでソート
    return df_after.sort_values(by="業者CD").reset_index(drop=True)
