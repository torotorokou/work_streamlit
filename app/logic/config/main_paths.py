# main_paths.py

from pathlib import Path
from logic.config.yaml_loader import YamlConfigLoader, YamlPathResolver


# --- 1. YAMLローダー（構造化された読み込み） ---
class MainPathsLoader:
    def __init__(self):
        # ここでYAMLファイルのパスを解決
        path_dict = {"main_paths": Path("config/paths/main_paths.yaml")}
        path_loader = YamlPathResolver(path_dict)
        self._loader = YamlConfigLoader(path_loader)

    def get_section(self, key: str) -> dict:
        data = self._loader.load_yaml_by_key("main_paths")
        return data.get(key, {})


# --- 2-1. 文字列→Path変換だけを担当するクラス ---
class PathConverter:
    @staticmethod
    def convert(section_data: dict) -> dict[str, Path]:
        return {k: Path(v) for k, v in section_data.items()}


# --- 2-2. Pathに変換されたデータを操作するクラス ---
class PathAccessor:
    def __init__(self, path_dict: dict[str, Path]):
        self._paths = path_dict

    def get(self, key: str) -> Path:
        return self._paths.get(key)

    def as_dict(self) -> dict[str, Path]:
        return self._paths

    def __repr__(self) -> str:
        lines = [f"  {k}: {v}" for k, v in self._paths.items()]
        return "PathAccessor:\n" + "\n".join(lines)


# --- 3. 全体をまとめて扱うファサードクラス ---
class MainPaths:
    def __init__(self):
        loader = MainPathsLoader()

        self.csv = PathAccessor(PathConverter.convert(loader.get_section("csv")))
        self.directories = PathAccessor(PathConverter.convert(loader.get_section("directories")))
        self.logs = PathAccessor(PathConverter.convert(loader.get_section("logs")))
        self.yaml_files = PathAccessor(PathConverter.convert(loader.get_section("yaml_files")))
