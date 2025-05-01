from logic.config.main_paths import MainPathsLoader
from logic.config.yaml_loader import (
    YamlConfigLoader,
    YamlPathResolver,
    YamlLoaderInterface,
)


class AppSettingLoader(YamlLoaderInterface):
    """main_paths.yaml 経由で app_setting.yaml を読み込むクラス"""

    def __init__(self):
        path_dict = MainPathsLoader().yaml_files.as_dict()
        self.loader = YamlConfigLoader(YamlPathResolver(path_dict))

    def get(self) -> dict:
        return self.loader.load("app_setting")
