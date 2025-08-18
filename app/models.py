from typing import Annotated, Optional, List
from pydantic import BaseModel, StringConstraints, Field

class QuestionRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(min_length=1, max_length=32768)
    ] = Field(
        description="The user's question to send directly to the LLM (no retrieval).",
        json_schema_extra={"example": "What is Retrieval-Augmented Generation?"}
    )

class IngestRequest(BaseModel):
    paths: Optional[List[str]] = Field(
        default=None,
        description="List of file or directory paths to ingest. If not provided, uses the default docs_dir.",
        json_schema_extra={"example": ["data/docs/sample.md"]}
    )

class IngestResponse(BaseModel):
    ingested_chunks: int = Field(
        description="Number of document chunks successfully ingested and indexed.",
        json_schema_extra={"example": 1}
    )

class ChatRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(min_length=1, max_length=32768)
    ] = Field(
        description="The user's question to be answered using the ingested documents.",
        json_schema_extra={"example": "What is Retrieval-Augmented Generation?"}
    )
    top_k: Annotated[
        int,
        Field(ge=1, le=20)
    ] = Field(
        default=4,
        description="Maximum number of top document chunks to retrieve for context (1-20).",
        json_schema_extra={"example": 4}
    )

class ChatSource(BaseModel):
    index: int = Field(
        description="Reference number used in the answer, starting from 1",
        json_schema_extra={"example": 1}
    )
    id: str = Field(
        description="Unique identifier of the retrieved chunk",
        json_schema_extra={"example": "data/docs/sample.md:0"}
    )
    source: str = Field(
        description="Original file path or source label",
        json_schema_extra={"example": "data/docs/sample.md"}
    )
    text: str = Field(
        description="The actual retrieved chunk text",
        json_schema_extra={"example": "RAG stands for Retrieval-Augmented Generation."}
    )

class ChatResponse(BaseModel):
    answer: str = Field(
        description="The LLM's final answer to the user's question",
        json_schema_extra={
            "example": (
                "Retrieval-Augmented Generation (RAG) is a technique that combines a "
                "retrieval system with a generative model to provide answers grounded in "
                "external documents."
            )
        }
    )
    sources: List[ChatSource] = Field(
        description="List of supporting source chunks (IDs and source paths only)",
        json_schema_extra={"example": [
            {
                "index": 1,
                "id": "data/docs/sample.md:0",
                "source": "data/docs/sample.md",
                "text": "RAG stands for Retrieval-Augmented Generation."
            }
        ]}
    )