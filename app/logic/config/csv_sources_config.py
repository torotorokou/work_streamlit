from logic.config.main_paths import MainPaths
from logic.config.yaml_loader import YamlConfigLoader,YamlPathResolver


class CsvSourcesConfig:
    """main_paths.yaml 経由で csv_sources.yaml を読み込むクラス"""

    def __init__(self):
        path_dict = MainPaths().yaml_files.as_dict()
        resolver = YamlPathResolver(path_dict)
        self.loader = YamlConfigLoader(resolver)

    def get(self) -> dict:
        return self.loader.load("csv_sources_config")
