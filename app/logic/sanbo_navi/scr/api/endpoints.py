from fastapi import APIRouter
from scr.api.models import QueryRequest, QueryResponse
from scr import ai_loader, loader

router = APIRouter()

@router.post("/generate-answer", response_model=QueryResponse)
async def generate_answer(request: QueryRequest):
    answer, sources = ai_loader.get_answer(request.query, request.category, request.tags)
    return QueryResponse(answer=answer, sources=sources)

@router.get("/question-options")
def get_question_options():
    paths = loader.get_resource_paths()
    data = loader.load_question_templates()
    grouped = loader.group_templates_by_category_and_tags(data)
    return grouped
