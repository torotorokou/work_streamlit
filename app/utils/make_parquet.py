import pandas as pd


def csv_to_parquet(csv_path: str, parquet_path: str) -> None:
    """
    指定されたCSVファイルを読み込み、Parquet形式で保存する。

    Parameters:
        csv_path (str): 読み込むCSVファイルのパス
        parquet_path (str): 保存するParquetファイルのパス
    """
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)


# 使い方例
if __name__ == "__main__":
    input_csv = "/work/app/data/input/出荷一覧_20250718_102956.csv"
    output_parquet = "/work/app/data/input/debug_shipping.parquet"
    csv_to_parquet(input_csv, output_parquet)
