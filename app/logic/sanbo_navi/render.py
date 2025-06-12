# --- ライブラリ読み込み ---
import streamlit as st
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # 修正: langchain_openaiからインポート


def load_config():
    """環境変数から設定を読み込みます。

    OPENAI_API_KEY が環境変数に設定されていない場合、OpenAIクライアントは None を返します。

    戻り値:
        tuple: OpenAIクライアント（または None）、PDFパス、JSONパス、FAISSパス、OpenAI APIキーを含むタプル。
    """

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # OpenAI APIキーの取得
    if not openai_api_key or len(openai_api_key) < 32 or not openai_api_key.isalnum():
        st.warning(
            "OPENAI_API_KEY が環境変数に設定されていないか、無効です。APIキーを正しく設定してください。"
        )
        st.warning(
            "OPENAI_API_KEY が環境変数に設定されていません。APIキーを設定してください。"
        )
        client = None
    else:
        client = OpenAI(api_key=openai_api_key)

    # パスの取得
    PDF_PATH = "data/SOLVEST.pdf"
    JSON_PATH = "structured_SOLVEST_output_final.json"
    FAISS_PATH = "vectorstore/solvest_faiss_corrected"

    return client, PDF_PATH, JSON_PATH, FAISS_PATH, openai_api_key


def render_education_gpt_page():
    # ここに今の Streamlit UI 部分をすべて入れる
    # st.title("📘 教育GPTアシスタント")
    st.markdown(
        "SOLVESTについて質問できます。まず最初に知りたいことを一言で入力してください。"
    )
    # --- 設定の読み込み ---
    client, PDF_PATH, JSON_PATH, FAISS_PATH, openai_api_key = load_config()

    # --- 初期設定 ---

    # --- PDF画像の読み込み ---
    @st.cache_resource
    def load_pdf_first_page(path, dpi=100):
        # PDFの1ページ目を画像として読み込む
        return convert_from_path(path, dpi=dpi, first_page=1, last_page=1)

    @st.cache_resource
    def load_pdf_page(path, page_number, dpi=100):
        # 指定ページのPDFを画像として読み込む
        return convert_from_path(
            path, dpi=dpi, first_page=page_number, last_page=page_number
        )[0]

    # --- ベクトルストア読み込み ---
    @st.cache_resource
    def load_vectorstore():
        # FAISSベクトルストアのロード
        if not openai_api_key:
            st.warning(
                "OPENAI_API_KEY が未設定のためベクトルストアをロードできません。"
            )
            return None
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        return FAISS.load_local(
            FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True
        )

    vectorstore = load_vectorstore()

    # --- JSONからカテゴリ・サブカテゴリを取得 ---
    @st.cache_data
    def load_json_data(json_path):
        # JSONファイルを読み込んでデータとして返す
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data

    json_data = load_json_data(JSON_PATH)

    @st.cache_data
    def extract_categories_and_titles(data):
        # JSONからカテゴリとサブカテゴリを抽出
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

    categories, subcategory_map = extract_categories_and_titles(json_data)

    # --- カテゴリ別 質問テンプレート ---
    category_question_templates = {
        # 各カテゴリに対して質問テンプレートを用意
        "処理工程": [
            "この工程の流れを教えて",
            "処理対象の廃棄物は？",
            "使われている設備は？",
            "安全対策や注意点は？",
            "処理能力は？",
        ],
        "設備": [
            "この設備の用途は？",
            "この設備の処理能力は？",
            "設備の仕様と特徴は？",
            "安全対策や注意点は？",
            "メンテナンス頻度は？",
        ],
        "行政・許認可": [
            "この施設に必要な許可は？",
            "許認可の申請手続きは？",
            "行政提出書類に何が必要？",
            "許認可取得の流れは？",
            "許可の更新や管理は？",
        ],
        "産廃分類・品目": [
            "この品目はどんな廃棄物？",
            "この品目の処理方法は？",
            "搬入時の注意点は？",
            "処理後の搬出先は？",
            "関連する設備は？",
        ],
        "施設": [
            "施設の維持管理項目は？",
            "点検頻度や管理方法は？",
            "設備ごとの管理内容は？",
            "管理記録はどのようにする？",
            "異常時の対応は？",
        ],
        "default": [
            "この工程の流れを教えて",
            "処理対象の廃棄物は？",
            "使われている設備は？",
            "処理能力は？",
            "安全対策や注意点は？",
        ],
    }

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
    category_template = category_question_templates.get(
        main_category, category_question_templates["default"]
    )
    subcategory_options = (
        ["自由入力"] + subcategory_map.get(main_category, []) + category_template
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
    render_education_gpt_page()
