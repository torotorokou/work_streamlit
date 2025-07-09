import os
import json
import yaml
from pathlib import Path

def get_resource_paths() -> dict:
    base_path = Path("/app")  # Dockerコンテナ内のルート
    return {
        "PDF_PATH": base_path / "local_data" / "master" / "SOLVEST.pdf",
        "JSON_PATH": base_path / "local_data" / "master" / "structured_SOLVEST_output_with_tags.json",
        "FAISS_PATH": base_path / "local_data" / "master" / "vectorstore" / "solvest_faiss_corrected",
        "ENV_PATH": base_path / "config" / ".env",
        "YAML_PATH": base_path / "local_data" / "master" / "category_question_templates_with_tags.yaml",
    }

def load_json_data(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

def load_question_templates():
    yaml_path = get_resource_paths().get("YAML_PATH")
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_categories_and_titles(data):
    categories = set()
    subcategories = {}
    for section in data:
        cats = section.get("category", [])
        if isinstance(cats, str):
            cats = [cats]
        for cat in cats:
            categories.add(cat)
            subcategories.setdefault(cat, set()).add(section.get("title"))
    categories = sorted(categories)
    for k in subcategories:
        subcategories[k] = sorted(subcategories[k])
    return categories, subcategories

def group_templates_by_category_and_tags(data):
    grouped = {}
    for section in data:
        category = section.get("category")
        tags = tuple(section.get("tags", []))
        title = section.get("title")
        grouped.setdefault(category, {}).setdefault(tags, []).append(title)
    return grouped
