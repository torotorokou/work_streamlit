import yaml
from pathlib import Path  # ← これを追加

# from logic.config.main_paths import MainPaths
from abc import ABC, abstractmethod
from typing import Any


class YamlConfigLoader:
    def __init__(self, path_loader):
        self.path_loader = path_loader

    def load_yaml_by_key(self, key: str) -> dict:
        path = self.path_loader.get_config_path(key)
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)


class YamlPathResolver:
    def __init__(self, path_dict: dict[str, Path]):
        self._path_dict = path_dict

    def get_path(self, key: str) -> Path:
        if key not in self._path_dict:
            raise KeyError(f"指定されたYAMLキーが存在しません: {key}")
        return self._path_dict[key]


class YamlLoaderInterface(ABC):
    @abstractmethod
    def get(self) -> Any:
        """YAMLの構造化データを返す"""
        pass
