from typing import Annotated, Optional, List
from pydantic import BaseModel, StringConstraints

class QuestionRequest(BaseModel):
    question: Annotated[str, StringConstraints(min_length=1, max_length=32768)]

class IngestRequest(BaseModel):
    paths: Optional[List[str]] = None
    
class IngestResponse(BaseModel):
    ingested_chunks: int
    
class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4