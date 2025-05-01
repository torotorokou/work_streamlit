from logic.config.templates_config import TemplatesConfig

class TemplateFileRequirementResolver:
    def __init__(self):
        self.config = TemplatesConfig().get()

    def required_files_map(self) -> dict[str, list[str]]:
        return {key: val.get("required_files", []) for key, val in self.config.items()}


class TemplateLabelResolver:
    def __init__(self):
        self.config = TemplatesConfig().get()

    def label_to_key_dict(self) -> dict[str, str]:
        return {val["label"]: key for key, val in self.config.items()}

    def key_to_label_dict(self) -> dict[str, str]:
        return {key: val["label"] for key, val in self.config.items()}

    def all_labels(self) -> list[str]:
        return [val["label"] for val in self.config.values()]


class TemplateConfigFacade:
    def __init__(self):
        self.config = TemplatesConfig().get()
        self.file_resolver = TemplateFileRequirementResolver()
        self.label_resolver = TemplateLabelResolver()

    def required_files_map(self):
        return self.file_resolver.required_files_map()

    def label_to_key_dict(self):
        return self.label_resolver.label_to_key_dict()

    def key_to_label_dict(self):
        return self.label_resolver.key_to_label_dict()
