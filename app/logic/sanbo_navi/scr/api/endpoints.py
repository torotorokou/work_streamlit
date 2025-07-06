from fastapi import APIRouter
from api.models import QueryRequest, QueryResponse
import ai_loader
import llm_utils
import loader
import utils

router = APIRouter()

@router.post("/generate-answer", response_model=QueryResponse)
async def generate_answer(request: QueryRequest):
    # 1. 必要なパス取得
    FAISS_PATH, PDF_PATH, JSON_PATH = loader.load_config()

    # 2. JSONデータ読み込み（タグ・カテゴリ付き構造データ）
    json_data = loader.load_json_data(JSON_PATH)

    # 3. AIクライアント取得
    client = ai_loader.load_ai(ai_loader.OpenAIConfig)
    llm_client = llm_utils.OpenAIClient(client)

    # 4. ベクトルストア読み込み
    vectorstore = utils.load_vectorstore(api_key=client.api_key, FAISS_PATH=FAISS_PATH)

    # 5. 回答生成
    answer, sources = llm_utils.generate_answer(
        query=request.query,
        selected_category=request.category,
        vectorstore=vectorstore,
        llm_client=llm_client,
    )

    return QueryResponse(answer=answer, sources=sources)
