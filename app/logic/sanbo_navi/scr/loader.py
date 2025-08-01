
import os
from dotenv import load_dotenv
import streamlit as st
import yaml
import json

def get_resource_paths() -> dict:
    return {
        "PDF_PATH": "/work/app/logic/sanbo_navi/local_data/master/SOLVEST.pdf",
        "PDF_PATH_PLAN": "/work/app/logic/sanbo_navi/local_data/master/solvest_business_plan_20240305.pdf",
        "JSON_PATH": "/work/app/logic/sanbo_navi/local_data/master/structured_SOLVEST_output_with_tags.json",
        "FAISS_PATH": "/work/app/logic/sanbo_navi/local_data/master/vectorstore/solvest_faiss_corrected",
        "ENV_PATH": "/work/app/logic/sanbo_navi/config/.env",
        "YAML_PATH": "/work/app/logic/sanbo_navi/local_data/master/category_question_templates.yaml",
    }

def load_config():
    paths = get_resource_paths()
    return paths["FAISS_PATH"], paths["PDF_PATH"], paths["PDF_PATH_PLAN"], paths["JSON_PATH"]

@st.cache_data
def load_question_templates():
    yaml_path = get_resource_paths().get("YAML_PATH")
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_json_data(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
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
    return sorted(categories), {k: sorted(v) for k, v in subcategories.items()}
