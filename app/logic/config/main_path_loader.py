from pathlib import Path
import yaml
import os


class MainPathLoader:
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or os.getenv("BASE_DIR", "/work/app"))
        self.main_paths = self._load_main_paths()

    def _load_main_paths(self) -> dict:
        path = self.base_dir / "config/paths/main_paths.yaml"
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_config_path(self, key: str) -> Path:
        return self.base_dir / self.main_paths["config_files"][key]

    def get_csv_path(self, key: str) -> Path:
        return self.base_dir / self.main_paths["csv"][key]
