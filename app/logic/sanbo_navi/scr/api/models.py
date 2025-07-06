from pydantic import BaseModel
from typing import List, Tuple

class QueryRequest(BaseModel):
    query: str
    category: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Tuple[str, str]]
