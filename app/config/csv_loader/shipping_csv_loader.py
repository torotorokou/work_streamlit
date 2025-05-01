# logic/data_loader/shipping_csv_loader.py

import pandas as pd
from logic.config.csv_loader import DataFrameLoaderInterface
from logic.config.main_paths import MainPathsLoader


class ShippingCsvLoader(DataFrameLoaderInterface):
    """main_paths.yaml で定義された shipping.csv を読み込むローダー"""

    def __init__(self):
        path = MainPathsLoader().csv.get("shipping")  # ← 例: config経由でPathを取得
        self._path = path

    def get(self) -> pd.DataFrame:
        return pd.read_csv(self._path, encoding="utf-8")
