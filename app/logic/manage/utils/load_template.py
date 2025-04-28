import pandas as pd


def load_master_and_template(master_path):
    """
    平均表テンプレート用のマスターCSVを読み込む。
    各列に対して明示的にデータ型を指定し、「値」列は object 型として混在データに対応。

    Parameters:
        config (dict): 設定情報（config["templates"]["average_sheet"]["master_csv_path"] を使用）

    Returns:
        pd.DataFrame: 型指定されたマスターCSVの内容
    """

    dtype_spec = {
        "大項目": str,
        "小項目1": str,
        "小項目2": str,
        "セル": str,
        "値": "object",  # 数値・日付・文字列 なんでも入る列
    }

    master_csv = pd.read_csv(master_path, encoding="utf-8-sig", dtype=dtype_spec)

    return master_csv
