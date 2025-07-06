from typing import List, Tuple, Optional
from scr.prompts import build_prompt
from scr.utils import search_documents_with_category
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../config/.env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query: str, category: str, json_data: List[dict], vectorstore, tags: Optional[List[str]] = None) -> dict:
    retrieved = search_documents_with_category(query, category, json_data, vectorstore, tags=tags)
    context = "\n".join([r[1] for r in retrieved])
    prompt = build_prompt(query, context)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "sources": retrieved}
