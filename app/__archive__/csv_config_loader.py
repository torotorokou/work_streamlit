from .base_loader import BaseConfigLoader


class CSVConfigLoader(BaseConfigLoader):
    def get_label_map(self) -> dict:
        config = self._load_yaml(self.paths["yaml_files"]["csv_sorces_config"])
        return {key: value["label"] for key, value in config.items()}
