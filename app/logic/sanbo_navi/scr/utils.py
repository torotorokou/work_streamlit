# ======= scr/utils.py =======
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional

def load_faiss_vectorstore(faiss_path):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)

def search_documents_with_category(
    query: str,
    category: str,
    json_data: List[dict],
    vectorstore,
    top_k: int = 4,
    tags: Optional[List[str]] = None
):
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    filtered = []
    for doc, score in results:
        meta = doc.metadata
        if meta.get("category") == category:
            if tags is None or any(tag in meta.get("tag", []) for tag in tags):  # ← 修正済
                filtered.append((meta.get("title", "Unknown"), doc.page_content))
    return filtered
