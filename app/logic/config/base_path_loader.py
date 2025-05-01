# utils/base/base_path_loader.py

from pathlib import Path
from abc import ABC, abstractmethod
from logic.config.main_paths import MainPaths


class BasePathLoader(ABC):
    """main_paths.yaml の directories セクションからパスを取得する基本クラス"""

    def __init__(self, key: str):
        self._key = key
        self._path = MainPaths().directories.get(key)

    def get(self) -> Path:
        return self._path

    def ensure_exists(self) -> None:
        """ディレクトリが存在しない場合は作成する"""
        self._path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._key} = {self._path}>"


class PathLoaderInterface(ABC):
    @abstractmethod
    def get(self) -> Path:
        """特定のファイルやディレクトリのパスを返す"""
        pass
