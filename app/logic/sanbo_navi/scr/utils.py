from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional
import ast


def load_faiss_vectorstore(faiss_path):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)


def safe_parse_tags(raw):
    """
    タグを常にリスト化して返す（例：文字列形式 '["設備", "PVC"]' → ["設備", "PVC"]）
    """
    if isinstance(raw, list):
        return raw
    elif isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
            else:
                return [parsed]
        except:
            return [raw]  # 例外時は文字列のまま1要素リストに
    return []


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
        doc_category_raw = meta.get("category")
        doc_tags_raw = meta.get("tag", [])

        # --- カテゴリ整形（str または list のどちらにも対応）
        if isinstance(doc_category_raw, list):
            doc_category = doc_category_raw[0] if doc_category_raw else ""
        else:
            doc_category = str(doc_category_raw)

        doc_tags = safe_parse_tags(doc_tags_raw)

        print("=== DEBUG ===")
        print("Category (parsed):", doc_category)
        print("Tags (parsed):", doc_tags)
        print("→ Matching tags:", tags)
        print(f"=== [INFO] Filtered documents: {len(filtered)}")

        if doc_category == category:
            if tags is None or any(tag in doc_tags for tag in tags):
                filtered.append((meta.get("title", "Unknown"), doc.page_content))

    return filtered
