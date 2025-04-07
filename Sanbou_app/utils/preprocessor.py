import pandas as pd

def process_csv_by_date(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    CSVデータを日付カラムでソートし、最小日付のデータのみ抽出する。

    Parameters:
        df (pd.DataFrame): 処理対象のDataFrame
        date_column (str): 日付列の名前（例："伝票日付"）

    Returns:
        pd.DataFrame: ソート＆フィルタ済みのDataFrame
    """
    # 日付変換（失敗したらNaTになる）
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # 有効な日付だけに絞る
    df = df.dropna(subset=[date_column])

    # 日付で昇順ソート
    df = df.sort_values(by=date_column)

    # 最小日付を取得
    min_date = df[date_column].min()

    # 最小日付の行のみ抽出
    filtered_df = df[df[date_column] == min_date]

    return filtered_df
