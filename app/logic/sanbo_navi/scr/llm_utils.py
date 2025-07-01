# logic
from logic.sanbo_navi.scr.prompts import (
    build_category_suggestion_prompt,
    build_answer_prompt,
)

# --- GPT/LLM 呼び出し クラス版 ---
from abc import ABC, abstractmethod


# --- 抽象基底クラス ---
class LLMClientBase(ABC):
    def __init__(self, client):
        self.client = client

    @abstractmethod
    def call(self, system_prompt, user_prompt):
        pass


# --- OpenAI 用クラス ---
class OpenAIClient(LLMClientBase):
    def __init__(self, client, model_name="gpt-4o"):
        super().__init__(client)
        self.model_name = model_name

    def call(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # --- タグ抽出用の簡易プロンプト ---
    def extract_tags(self, query: str) -> list:
        tag_extraction_prompt = (
            f"次の質問に関連するタグ候補を最大5個、箇条書きで挙げてください: {query}"
        )
        raw = self.call(
            "あなたは廃棄物処理の専門AIです。タグ候補を抽出してください。",
            tag_extraction_prompt,
        )
        tags = [
            line.strip("- ・* ").strip() for line in raw.split("\n") if line.strip()
        ]
        return tags[:5]


# --- 回答生成 ---
def generate_answer(query, selected_category, vectorstore, llm_client: OpenAIClient):
    # ステップ1: ドキュメント検索（MMR）
    docs = vectorstore.max_marginal_relevance_search(query, k=10, fetch_k=30)

    # ステップ2: タグ抽出（安全に）
    try:
        candidate_tags = llm_client.extract_tags(query)
    except Exception:
        candidate_tags = []

    # ステップ3: スコア付け（カテゴリ・タグに一致するほど優先）
    def score(doc):
        score = 0
        category_list = doc.metadata.get("category", [])
        tag_list = doc.metadata.get("tag", [])

        if selected_category in category_list:
            score += 2
        score += sum(1 for tag in candidate_tags if tag in tag_list)

        return score

    scored_docs = sorted(docs, key=score, reverse=True)
    top_docs = scored_docs[:5]

    # ステップ4: コンテキスト生成
    context = "\n".join([doc.page_content for doc in top_docs])
    sources = [
        (doc.metadata.get("source", "不明"), doc.metadata.get("page", "不明"))
        for doc in top_docs
    ]

    # ステップ5: プロンプト構築＋LLM呼び出し
    system_prompt = "あなたは日本の産業廃棄物処理に関する専門的な情報提供AIです。以下の文書を参考に、できるだけ正確に答えてください。"
    user_prompt = (
        f"以下の文書情報を参考に質問に答えてください:\n{context}\n\n質問:{query}"
    )

    answer = llm_client.call(system_prompt, user_prompt)
    return answer, sources
