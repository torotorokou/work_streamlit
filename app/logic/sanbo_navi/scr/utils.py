from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from typing import List, Optional

def load_faiss_vectorstore(faiss_path):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)

def search_documents_with_category(query, category, json_data, vectorstore, top_k=4, tags: Optional[List[str]] = None):
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    filtered = []
    for doc, score in results:
        meta = doc.metadata
        if meta.get("category") == category:
            if tags is None or any(tag in meta.get("tags", []) for tag in tags):
                filtered.append((meta.get("title", "Unknown"), doc.page_content))
    return filtered