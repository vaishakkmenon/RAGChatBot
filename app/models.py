from typing import Annotated, Optional, List
from pydantic import BaseModel, StringConstraints, Field

class QuestionRequest(BaseModel):
    question: Annotated[str, StringConstraints(min_length=1, max_length=32768)]

class IngestRequest(BaseModel):
    paths: Optional[List[str]] = None
    
class IngestResponse(BaseModel):
    ingested_chunks: int
    
class ChatRequest(BaseModel):
    question: Annotated[str, StringConstraints(min_length=1, max_length=32768)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 4