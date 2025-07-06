from pydantic import BaseModel
from typing import List, Tuple, Optional

class QueryRequest(BaseModel):
    query: str
    category: str
    tags: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Tuple[str, str]]