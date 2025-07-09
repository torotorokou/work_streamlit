from scr.llm_utils import generate_answer
from scr.loader import get_resource_paths, load_json_data
from scr.utils import load_faiss_vectorstore
from typing import List, Optional

def get_answer(query: str, category: str, tags: Optional[List[str]] = None):
    paths = get_resource_paths()
    json_data = load_json_data(paths["JSON_PATH"])
    vectorstore = load_faiss_vectorstore(paths["FAISS_PATH"])
    result = generate_answer(query, category, json_data, vectorstore, tags)
    return result["answer"], result["sources"]