from logic.config.main_paths import MainPaths
from logic.config.yaml_loader import YamlConfigLoader, YamlPathResolver,YamlLoaderInterface


class TemplatesConfig(YamlLoaderInterface):
    """main_paths.yaml 経由で templates_config.yaml を読み込むクラス"""

    def __init__(self):
        path_dict = MainPaths().yaml_files.as_dict()
        resolver = YamlPathResolver(path_dict)
        self.loader = YamlConfigLoader(resolver)

    def get(self) -> dict:
        return self.loader.load("templates_config")
