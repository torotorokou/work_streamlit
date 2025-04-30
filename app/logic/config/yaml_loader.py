import yaml

class YamlConfigLoader:
    def __init__(self, path_loader):
        self.path_loader = path_loader

    def load_yaml_by_key(self, key: str) -> dict:
        path = self.path_loader.get_config_path(key)
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
