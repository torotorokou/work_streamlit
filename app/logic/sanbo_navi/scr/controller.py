# --- ライブラリ読み込み ---
import streamlit as st
import json
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from logic.sanbo_navi.scr.view import load_pdf_first_page, load_pdf_page


from logic.sanbo_navi.scr.loader import (
    load_config,
    load_json_data,
    extract_categories_and_titles,
)
from logic.sanbo_navi.scr.loader import load_question_templates
from logic.sanbo_navi.scr.loader import get_resource_paths

# AI設定
from logic.sanbo_navi.scr.ai_loader import OpenAIConfig
from logic.sanbo_navi.scr.ai_loader import load_ai

# 外部読込
from logic.sanbo_navi.scr.loader import get_resource_paths


# --- ベクトルストアの読み込み ---
@st.cache_resource
def load_vectorstore(api_key: str = None, FAISS_PATH: str = None):
    # FAISSベクトルストアのロード
    if not api_key:
        st.warning("OPENAI_API_KEY が未設定のためベクトルストアをロードできません。")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.load_local(
        FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True
    )


def contoroller_education_gpt_page():
    # st.title("📘 教育GPTアシスタント")
    # --- 設定の読み込み ---
    FAISS_PATH, PDF_PATH, JSON_PATH = load_config()

    #  AIの設定を読込
    client = load_ai(OpenAIConfig)  # OpenAIConfigをデフォルトクラスとして使用

    # ベクトルストアの読込
    vectorstore = load_vectorstore(api_key=client.api_key, FAISS_PATH=FAISS_PATH)

    # --- JSONデータの読込 ---
    json_data = load_json_data(JSON_PATH)
    categories, subcategory_map = extract_categories_and_titles(json_data)

    # --- カテゴリsuggestion用関数 ---
    def suggest_category(query_input: str) -> str:
        # GPTを使ってユーザーの質問から最適なカテゴリを提案する
        prompt = f"""
    以下は、日本の産業廃棄物処理に関する教育用AIシステムで使われるカテゴリの一覧です：
    
    - 処理工程
    - 設備
    - 行政・許認可
    - 産廃分類・品目
    - 施設
    
    次の質問は、どのカテゴリに分類するのがもっとも適切でしょうか？
    カテゴリ名を1つだけ出力してください。（それ以外は出力しないでください）
    
    【質問】  
    {query_input}
    """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "あなたは日本の産業廃棄物処理のカテゴリ分類に詳しい専門アシスタントです。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        suggestion = response.choices[0].message.content.strip()
        return suggestion

    # --- UI構築 ---
    # --- タイトル・説明 ---
    st.title("📘 教育GPTアシスタント")
    st.markdown(
        "SOLVESTについて質問できます。まず最初に知りたいことを一言で入力してください。"
    )

    # --- PDF 1ページ目 プレビュー ---
    with st.expander("📄 PDFプレビュー（1ページ目のみ先に表示）"):
        pdf_first_page = load_pdf_first_page(PDF_PATH)
        st.image(pdf_first_page[0], caption="Page 1", use_column_width=True)

    # --- ステップ① ユーザー自由入力 ---
    query_input = st.text_input(
        "知りたいこと（例：六面梱包機の処理能力、許認可の取得方法 など）"
    )

    # --- ステップ② カテゴリ自動提案 ---
    suggested_category = None
    if st.button("➡️ カテゴリを自動提案"):
        if query_input.strip():
            with st.spinner("🤖 カテゴリを提案中..."):
                suggested_category = suggest_category(query_input)
                st.session_state.suggested_category = suggested_category
        else:
            st.warning("まず知りたいことを入力してください。")

    # --- ステップ③ カテゴリ選択box ---
    default_category = st.session_state.get("suggested_category", None)
    main_category = st.selectbox(
        "カテゴリを選択",
        options=categories,
        index=(
            categories.index(default_category) if default_category in categories else 0
        ),
    )

    # --- ステップ④ 対象選択 ---
    # 質問テンプレートのロード
    templates = load_question_templates()
    category_template = templates.get(main_category, templates["default"])

    # サブカテゴリ選択肢の生成
    subcategory_options = ["自由入力"] + templates.get(
        main_category, templates["default"]
    )
    sub_category = st.selectbox("対象を選択", options=subcategory_options)

    # --- 質問文構築 ---
    if sub_category == "自由入力":
        user_input = st.text_area("さらに詳しく質問を入力", height=100)
        query = user_input if user_input.strip() else query_input
    else:
        query = (
            f"{query_input} に関して、{sub_category} を詳しく教えてください。"
            if query_input.strip()
            else f"{sub_category} に関して、詳しく教えてください。"
        )

    # --- 回答生成 ---
    def generate_answer(query: str, selected_category: str):
        # ベクトルストア検索で関連文書取得
        docs = vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=30)

        # 選択カテゴリがある場合はフィルタリング
        if selected_category:
            docs = [
                doc
                for doc in docs
                if selected_category in doc.metadata.get("category", [])
            ]

        # コンテキスト作成
        context = "\n".join([doc.page_content for doc in docs])
        sources = [
            (doc.metadata.get("source", "不明"), doc.metadata.get("page", "不明"))
            for doc in docs
        ]

        # 回答生成用プロンプト
        prompt = f"""
    以下は、産業廃棄物の処理・設備・工程に関する構造化技術文書の抜粋です。
    この抜粋に基づき、以下の質問に対して **できる限り文書内容を優先しながら、工程順・論理順に沿って正確かつ実務的に**答えてください。
    
    【資料抜粋】
    {context}
    
    【質問】
    {query}
    
    ### 【回答ルール】
    1. 資料の記載に基づいて回答してください。
    2. 明記がない場合は一般的見解で補足してください。
    3. 回答は 1. 2. 3. のように箇条書きで。
    4. 信頼度や出典（ページ番号）も可能な限り明示してください。
    """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "あなたは日本の産業廃棄物処理文書をもとに回答する専門AIです。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content, sources

    # --- 回答表示 ---
    if st.button("➡️ 送信") and query.strip():
        with st.spinner("🤖 回答生成中..."):
            answer, sources = generate_answer(query, main_category)
            st.session_state.last_response = answer
            st.session_state.sources = sources

    if "last_response" in st.session_state:
        st.success("✅ 回答")
        st.markdown(st.session_state.last_response)

        if "sources" in st.session_state:
            # PDFページキャッシュの初期化
            if "cache_pdf_pages" not in st.session_state:
                st.session_state.cache_pdf_pages = {}

            # 出典ページ番号の抽出
            pages = set(int(p) for _, p in st.session_state.sources if str(p).isdigit())
            st.markdown("📄 **出典ページ:** " + ", ".join([f"Page {p}" for p in pages]))

            # 出典ページのプレビュー表示
            with st.expander("📘 出典ページのプレビュー"):
                for p in sorted(pages):
                    if isinstance(p, int) and p >= 1:
                        if p in st.session_state.cache_pdf_pages:
                            # キャッシュから読み込み
                            st.image(
                                st.session_state.cache_pdf_pages[p],
                                caption=f"Page {p} (cached)",
                                use_column_width=True,
                            )
                        else:
                            # PDFから読み込み
                            with st.spinner(f"📄 Page {p} 読み込み中..."):
                                page_image = load_pdf_page(PDF_PATH, p)
                                st.session_state.cache_pdf_pages[p] = page_image
                                st.image(
                                    page_image,
                                    caption=f"Page {p}",
                                    use_column_width=True,
                                )


if __name__ == "__main__":
    contoroller_education_gpt_page()
