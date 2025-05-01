# logic/config/csv_sources/csv_sources_resolver.py

from logic.config.csv_sources_config import CsvSourcesConfig


class CsvSourcesResolver:
    """csv_sources.yaml の構造変換ロジックを提供するクラス"""

    def __init__(self):
        self.config = CsvSourcesConfig().get()

    def label_map(self) -> dict:
        return {key: val["label"] for key, val in self.config.items()}

    def reverse_label_map(self) -> dict:
        return {val["label"]: key for key, val in self.config.items()}


class CsvSourcesResolver:
    def __init__(self):
        self.config = CsvSourcesConfig().get()

    def label_map(self) -> dict:
        return {key: val["label"] for key, val in self.config.items()}

    def date_column_map(self) -> dict:
        return {key: val["date_column"] for key, val in self.config.items()}
