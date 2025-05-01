import pandas as pd
from abc import ABC, abstractmethod


class CsvLoader:
    def __init__(self, path_loader):
        self.path_loader = path_loader

    def load_csv_by_key(self, key: str) -> pd.DataFrame:
        path = self.path_loader.get_csv_path(key)
        return pd.read_csv(path, encoding="utf-8-sig")


class DataFrameLoaderInterface(ABC):
    @abstractmethod
    def get(self) -> pd.DataFrame:
        """ファイルを読み込んでDataFrameを返す"""
        pass