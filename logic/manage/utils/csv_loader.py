

def load_filtered_dataframe(dfs, key, target_columns):
    """
    指定された辞書型DataFrameから、対象キーのDataFrameを取得し、指定されたカラムのみを抽出して返す。

    Parameters:
        dfs (dict): 複数のDataFrameを格納した辞書。例: {"receive": df1, "yard": df2}
        key (str): 対象となるDataFrameのキー名。例: "receive"
        target_columns (list): 抽出するカラム名のリスト。例: ["伝票日付", "品名", "正味重量"]

    Returns:
        pd.DataFrame: 指定されたカラムのみを持つDataFrame（フィルタ済み）

    Raises:
        KeyError: 指定されたキーがdfsに存在しない場合
        ValueError: 指定カラムの一部がDataFrameに存在しない場合
    """
    if key not in dfs:
        raise KeyError(f"{key} はdfsに存在しません。")

    df = dfs[key]
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{key} に次のカラムが存在しません: {missing_cols}")

    return df[target_columns]

