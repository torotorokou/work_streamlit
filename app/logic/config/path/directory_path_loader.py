from logic.config.base_path_loader import BasePathLoader, PathLoaderInterface


class OutputPathLoader(BasePathLoader, PathLoaderInterface):
    """main_paths.yaml の output ディレクトリのパスを返すローダー"""

    def __init__(self):
        super().__init__("output")


class TemplatePathLoader(BasePathLoader, PathLoaderInterface):
    """main_paths.yaml の output ディレクトリのパスを返すローダー"""

    def __init__(self):
        super().__init__("template")


class TempPathLoader(BasePathLoader, PathLoaderInterface):
    """main_paths.yaml の output ディレクトリのパスを返すローダー"""

    def __init__(self):
        super().__init__("temp")
