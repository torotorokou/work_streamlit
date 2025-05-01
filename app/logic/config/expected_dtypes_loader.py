# logic/config/expected_dtypes_loader.py

from logic.config.main_paths import MainPaths
from logic.config.yaml_loader import YamlConfigLoader,YamlPathResolver
from utils.type_converter import resolve_dtype  # dtype変換ユーティリティ関数を想定


class ExpectedDtypesLoader:
    """expected_dtypes.yaml の読み込みクラス"""

    def __init__(self):
        path_dict = MainPaths().yaml_files.as_dict()
        resolver = YamlPathResolver(path_dict)
        self.loader = YamlConfigLoader(resolver)

    def load(self) -> dict:
        return self.loader.load("expected_dtypes")


class ExpectedDtypesResolver:
    """expected_dtypes.yaml の構造変換（dtype解決）クラス"""

    def __init__(self):
        self.raw_yaml = ExpectedDtypesLoader().load()

    def resolve(self) -> dict:
        resolved = {}
        for template_key, file_map in self.raw_yaml.items():
            resolved[template_key] = {}
            for file_key, dtype_map in file_map.items():
                resolved[template_key][file_key] = {
                    col: resolve_dtype(dtype_str) for col, dtype_str in dtype_map.items()
                }
        return resolved
