from logic.config.main_paths import MainPaths
from logic.config.yaml_loader import YamlConfigLoader,YamlPathResolver


class AppSettingLoader:
    """main_paths.yaml 経由で app_setting.yaml を読み込むクラス"""

    def __init__(self):
        path_dict = MainPaths().yaml_files.as_dict()
        self.loader = YamlConfigLoader(YamlPathResolver(path_dict))

    def get(self) -> dict:
        return self.loader.load("app_setting")