from logic.config.main_paths import MainPaths
from logic.config.yaml_loader import YamlConfigLoader,YamlPathResolver



class CsvRequiredColumnsLoader:
    """csv_required_columns.yaml を読み込む I/Oクラス"""

    def __init__(self):
        path_dict = MainPaths().yaml_files.as_dict()
        resolver = YamlPathResolver(path_dict)
        self.loader = YamlConfigLoader(resolver)

    def load(self) -> dict:
        return self.loader.load("csv_required_columns")


class CsvRequiredColumnsResolver:
    """テンプレート名に応じて必要なCSV列定義を解決するクラス"""

    def __init__(self):
        self._all_defs = CsvRequiredColumnsLoader().load()

    def get(self, template_name: str) -> dict:
        return self._all_defs.get(template_name, {})
