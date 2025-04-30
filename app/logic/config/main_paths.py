# main_paths.py

from pathlib import Path
from utils.config_loader import load_yaml


# --- 1. YAMLローダー（読み込みだけ担当） ---
class MainPathsLoader:
    def __init__(self, path: str | Path):
        self._raw_data = load_yaml(path)

    def get_section(self, key: str) -> dict:
        return self._raw_data.get(key, {})


# --- 2-1. 文字列→Path変換だけを担当するクラス（SRPに沿った分離） ---
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


# --- 3. 全体をまとめて扱うファサードクラス ---
class MainPaths:
    def __init__(self, config_path: str | Path = "config/paths/main_paths.yaml"):
        loader = MainPathsLoader(config_path)

        self.csv = PathAccessor(PathConverter.convert(loader.get_section("csv")))
        self.directories = PathAccessor(PathConverter.convert(loader.get_section("directories")))
        self.logs = PathAccessor(PathConverter.convert(loader.get_section("logs")))
        self.config_files = PathAccessor(PathConverter.convert(loader.get_section("config_files")))

    def get(self, category: str, key: str) -> Path:
        section = getattr(self, category, None)
        if section is None:
            raise ValueError(f"カテゴリー {category} は存在しません")
        return section.get(key)

    def as_dict(self) -> dict:
        return {
            "csv": self.csv.as_dict(),
            "directories": self.directories.as_dict(),
            "logs": self.logs.as_dict(),
            "config_files": self.config_files.as_dict(),
        }
