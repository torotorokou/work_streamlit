import pandas as pd
from logic.config.csv_loader import DataFrameLoaderInterface
from logic.config.main_paths import MainPaths


class ReceiveHeaderDefinitionLoader(DataFrameLoaderInterface):
    """main_paths.yaml で定義された receive_header_definition.csv を読み込むローダー"""

    def __init__(self):
        self._path = MainPaths().csv.get("receive_header_definition")

    def get(self) -> pd.DataFrame:
        return pd.read_csv(self._path, encoding="utf-8")
