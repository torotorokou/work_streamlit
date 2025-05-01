from logic.config.base_path_loader import BasePathLoader, PathLoaderInterface
from pathlib import Path
from abc import ABC, abstractmethod
from logic.config.main_paths import MainPathsLoader


class BasePathLoader(ABC):
    """main_paths.yaml の directories セクションからパスを取得する基本クラス"""

    def __init__(self, key: str):
        self._key = key
        self._path = MainPathsLoader().directories.get(key)

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


class OutputPathLoader(BasePathLoader, PathLoaderInterface):
    """main_paths.yaml の output ディレクトリのパスを返すローダー"""

    def __init__(self):
        super().__init__("output")


class TemplatePathLoader(BasePathLoader, PathLoaderInterface):
    """main_paths.yaml の template ディレクトリのパスを返すローダー"""

    def __init__(self):
        super().__init__("template")


class TempPathLoader(BasePathLoader, PathLoaderInterface):
    """main_paths.yaml の temp ディレクトリのパスを返すローダー"""

    def __init__(self):
        super().__init__("temp")
