from logic.config.page_config.page_config_loader import PageConfigLoader


class PageConfigResolver:
    """
    page_config.yaml の構造変換・逆引き辞書・ラベルリストなどを提供するユーティリティクラス
    """

    def __init__(self):
        self.pages = PageConfigLoader().get()

    def to_label_id_dict(self) -> dict:
        return {p["label"]: p["id"] for p in self.pages}

    def to_id_label_dict(self) -> dict:
        return {p["id"]: p["label"] for p in self.pages}

    def labels(self) -> list[str]:
        return [p["label"] for p in self.pages]

    def get_all(self) -> tuple[dict, dict, list]:
        label_id = self.to_label_id_dict()
        id_label = {v: k for k, v in label_id.items()}
        labels = list(label_id.keys())
        return label_id, id_label, labels
