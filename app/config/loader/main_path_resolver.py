# app/config/loader/main_path_resolver.py

from pathlib import Path
import yaml
import os
from typing import Optional


class MainPathResolver:
    """
    main_paths.yaml を読み込んで、パスの解決を行う専用クラス。
    """

    def __init__(self, config_path: str = "config/main_paths.yaml", base_dir: Optional[str] = None):
        # --- BASEディレクトリの決定 ---
        env_base = os.getenv("BASE_DIR")
        self.base_dir = Path(base_dir or env_base or "/work/app")

        # --- YAMLファイルの絶対パス ---
        self.yaml_path = self.base_dir / config_path

        # --- YAMLの読み込み ---
        self._config_data = self._load_yaml()



    def _load_yaml(self) -> dict:
        with open(self.yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f)


    def get_config(self) -> dict:
        """main_paths.yaml の辞書全体を取得"""
        return self._config_data


    def get_path(self, key_or_path: str, section: Optional[str] = None) -> Path:
        """
        main_paths.yaml のキー、または任意の相対パスから絶対パスを解決する。

        Parameters:
            key_or_path (str): セクション内のキー or 相対パス
            section (str, optional): セクション名（例: 'csv'）

        Returns:
            Path: 絶対パス
        """
        if section:
            rel_path = self._config_data.get(section, {}).get(key_or_path)
            if rel_path is None:
                raise KeyError(f"'{section}.{key_or_path}' は {self.yaml_path.name} に存在しません")
            return self.base_dir / rel_path
        else:
            return self.base_dir / key_or_path
