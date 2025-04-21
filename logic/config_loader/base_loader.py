class BaseConfigLoader:
    def __init__(self, path_config_file="config/main_paths.yaml"):
        self.paths = self.load_yaml(path_config_file)

    def _load_yaml(self, path: str) -> dict:
        from pathlib import Path
        import yaml

        with open(Path(path), encoding="utf-8") as f:
            return yaml.safe_load(f)
