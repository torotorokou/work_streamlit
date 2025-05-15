# app/config/loader/main_path_resolver.py

from pathlib import Path
import yaml
import os
from typing import Optional,Union


class MainPath:
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


    def get_path(self, keys: Union[str, list[str]], section: Optional[str] = None) -> Path:
        """
        任意の階層のキーをたどって、パスを取得。

        Parameters:
            keys (str or list[str]): 取得したいキー、またはネストされたキーのリスト
            section (str, optional): 最初のセクション（省略時はkeysだけで探索）

        Returns:
            Path: 絶対パス
        """
        target = self._config_data

        if section:
            target = target.get(section, {})
            if target is None:
                raise KeyError(f"セクション '{section}' が見つかりません")

        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            target = target.get(key)
            if target is None:
                raise KeyError(f"キー '{'.'.join(keys)}' が見つかりません")

        return self.base_dir / target
