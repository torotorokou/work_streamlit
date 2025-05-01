from logic.config.main_paths import MainPaths
from logic.config.yaml_loader import YamlConfigLoader,YamlPathResolver


class PageConfigLoader:
    """page_config.yaml 全体を読み込むクラス"""

    def __init__(self):
        path_dict = MainPaths().yaml_files.as_dict()
        resolver = YamlPathResolver(path_dict)
        self.loader = YamlConfigLoader(resolver)

    def load(self) -> dict:
        return self.loader.load("page_config")
