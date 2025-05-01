from logic.config.yaml_loader.expected_dtypes_loader import ExpectedDtypesLoader


class ExpectedDtypesResolver:
    """expected_dtypes.yaml に対する構造変換・抽出を担う Resolver"""

    def __init__(self):
        self.config = ExpectedDtypesLoader().get()  # 全体を辞書として取得済み

    def get_by_template(self, template_key: str) -> dict:
        """
        指定テンプレートに対応する dtype 定義を返す
        """
        return self.config.get(template_key, {})
