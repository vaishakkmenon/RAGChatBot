from typing import Annotated, Optional, List
from pydantic import BaseModel, StringConstraints, Field

class QuestionRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(min_length=1, max_length=32768)
    ] = Field(
        description="The user's question to send directly to the LLM (no retrieval)."
    )

class IngestRequest(BaseModel):
    paths: Optional[List[str]] = Field(
        default=None,
        description="List of file or directory paths to ingest. If not provided, uses the default docs_dir."
    )

class IngestResponse(BaseModel):
    ingested_chunks: int = Field(
        description="Number of document chunks successfully ingested and indexed."
    )

class ChatRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(min_length=1, max_length=32768)
    ] = Field(
        description="The user's question to be answered using the ingested documents."
    )
    top_k: Annotated[
        int,
        Field(ge=1, le=20)
    ] = Field(
        4,
        description="Maximum number of top document chunks to retrieve for context (1-20)."
    )