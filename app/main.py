import os, asyncio, time
import ollama
import socket
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from .settings import settings
from .ingest import ingest_paths
from .retrieval import search, get_sample_chunks
from .middleware.logging import LoggingMiddleware
from .middleware.max_size import MaxSizeMiddleware
from .models import (
    QuestionRequest,
    IngestRequest,
    IngestResponse,
    ChatRequest,
    ChatResponse,
    ChatSource,
)

import logging
logger = logging.getLogger(__name__)

_CLIENT = ollama.Client(host=settings.ollama_host)
_MODEL = settings.ollama_model
_NUM_CTX = settings.num_ctx
REQUEST_TIMEOUT_S = settings.ollama_timeout
MAX_BYTES = settings.max_bytes

app = FastAPI(
    title="RAGChatBot (Local $0)",
    description="A fully local Retrieval-Augmented Generation chatbot. Ingest local docs and query them using Ollama LLM and ChromaDB, with source citation and tight OpenAPI docs.",
    version="0.2.1",
    summary="Self-hosted RAG chatbot using Ollama, SentenceTransformers, and ChromaDB."
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(MaxSizeMiddleware, max_bytes=MAX_BYTES)

@app.get(
    "/",
    summary="Health and available endpoints",
    response_description="API status and supported endpoints.",
    tags=["Health"],
)
def root():
    """Returns API status and available endpoints."""
    return {"ok": True, "message": "Hello from RAGChatBot"}

@app.post(
    "/chat-test",
    summary="Quick LLM test (no retrieval)",
    response_description="Direct LLM answer for a test prompt.",
    tags=["LLM"],
)
async def chat_test(req: QuestionRequest):
    """
    Test prompt for direct LLM connection. Does **not** use retrieval/augmentation.
    """
    prompt = req.question
    start = time.perf_counter()

    async def _call_chat():
        return await run_in_threadpool(
            _CLIENT.chat,
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": _NUM_CTX},
        )

    try:
        resp = await asyncio.wait_for(_call_chat(), timeout=REQUEST_TIMEOUT_S)
        answer = resp.get("message", {}).get("content", "")
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return {"ok": True, "answer": answer, "elapsed_ms": elapsed_ms}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Upstream timeout after {REQUEST_TIMEOUT_S}s")
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
            httpx.TransportError, OSError, socket.gaierror, socket.timeout,
            ConnectionResetError, BrokenPipeError) as e:
        raise HTTPException(status_code=503, detail="Ollama unreachable") from e
    except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=502, detail="Upstream error") from e

@app.get(
    "/health/ollama",
    summary="Check LLM server and model availability",
    response_description="Ollama health, configured model status, and timing info.",
    tags=["Health"],
)
async def ollama_health():
    """
    Checks connectivity to the Ollama LLM server and confirms the selected model is available.
    """
    start = time.perf_counter()
    try:
        data = await asyncio.wait_for(
            run_in_threadpool(lambda: _CLIENT.list()),
            timeout=REQUEST_TIMEOUT_S
        )
        models = data.get("models", []) if isinstance(data, dict) else []
        model_ready = any(isinstance(m, dict) and m.get("name") == _MODEL for m in models)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return {
            "ok": True,
            "host": os.getenv("OLLAMA_HOST", ""),
            "model": _MODEL,
            "model_ready": bool(model_ready),
            "num_ctx": _NUM_CTX,
            "elapsed_ms": elapsed_ms,
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Upstream timeout after {REQUEST_TIMEOUT_S}s")
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
            httpx.TransportError, OSError, socket.gaierror, socket.timeout,
            ConnectionResetError, BrokenPipeError) as e:
        raise HTTPException(status_code=503, detail="Ollama unreachable") from e
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail="Upstream error") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream error") from e

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into the retrieval database",
    response_description="Number of text chunks ingested from documents.",
    tags=["Documents"],
)
async def ingest_data(req: IngestRequest):
    """
    Ingest Markdown or plain text files from local paths (file or directory). Chunks are indexed for later retrieval.
    """
    paths = req.paths
    try:
        doc_len = ingest_paths(paths)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"File not found: {e}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=f"Permission error: {e}")
    except Exception as e:
        logger.exception("Error during ingestion")
        raise HTTPException(status_code=500, detail="Unexpected error during ingestion")
    logger.info(f"Ingested {doc_len} chunks from {paths or settings.docs_dir}")
    return IngestResponse(ingested_chunks=doc_len)

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question over your ingested documents",
    response_description="LLM answer with supporting source chunks.",
    tags=["Chat"],
)
async def chat(
    req: ChatRequest,
    max_distance: float = Query(
        0.65,
        ge=0.0, le=2.0,
        description=(
            "Maximum vector distance allowed for retrieved chunks. "
            "Lower = closer match, higher = looser match. "
            "Typical values: 0.5-0.8 for cosine distance."
        )
    )
):
    """
    Retrieval-augmented question answering: finds the most relevant chunks, sends as context to LLM, and returns the answer with citations.
    """
    user_question = req.question
    top_k = req.top_k if req.top_k is not None else settings.top_k
    retrieved_chunks = search(user_question, top_k, max_distance)

    numbered_context = []
    for i, match in enumerate(retrieved_chunks, start=1):
        numbered_context.append(f"[{i}] {match['source']}\n{match['text']}")
    context = "\n\n".join(numbered_context)

    llm_prompt = (
        "You are a concise assistant. Answer the question using ONLY the information below.\n"
        "Cite sources in your answer using the reference numbers like [1], [2]. "
        "If the answer is not contained in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n\n"
        "Answer:"
    )

    async def _call_chat():
        return await run_in_threadpool(
            _CLIENT.chat,
            model=settings.ollama_model,
            messages=[{"role": "user", "content": llm_prompt}],
            options={"num_ctx": settings.num_ctx},
        )

    try:
        resp = await asyncio.wait_for(_call_chat(), timeout=settings.ollama_timeout)
        answer = resp.get("message", {}).get("content", "")
        return ChatResponse(
            answer=answer,
            sources=[
                ChatSource(
                    index=i,
                    id=match["id"],
                    source=match["source"],
                    text=match["text"],
                )
                for i, match in enumerate(retrieved_chunks, start=1)
            ],
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Upstream timeout after {settings.ollama_timeout}s")
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
            httpx.TransportError, OSError, socket.gaierror, socket.timeout,
            ConnectionResetError, BrokenPipeError) as e:
        raise HTTPException(status_code=503, detail="Ollama unreachable") from e
    except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=502, detail="Upstream error") from e

@app.get(
    "/debug-search",
    summary="Debug: Retrieve raw chunks (no LLM)",
    response_description="List of matching chunks for a query.",
    tags=["Debug"],
)
def debug_search(
    q: str = Query(..., description="Search query string", min_length=1),
    k: int = Query(4, description="Number of top results to return", ge=1, le=20),
    max_distance: float = Query(
        0.65, ge=0.0, le=2.0,
        description=(
            "Maximum vector distance allowed for retrieved chunks. "
            "Lower = closer match, higher = looser match."
        )
    )
):
    """
    Directly query the retrieval database to see which chunks would be retrieved for a given string.
    """
    results = search(q, k, max_distance)
    logger.info(f"Debug-search: q='{q}', k={k}, distance={max_distance}, matches={len(results)}")
    return {
        "matches": [
            {
                "id": r["id"],
                "source": r["source"],
                "text": r["text"][:200] + ("..." if len(r["text"]) > 200 else "")
            }
            for r in results
        ],
        "count": len(results)
    }

@app.get(
    "/debug-ingest",
    summary="Debug: Show random sample chunks",
    response_description="List of sample chunks and their count.",
    tags=["Debug"],
)
def debug_ingest(
    n: int = Query(
        10, ge=1, le=50, description="Number of sample chunks to show (1-50)"
    )
):
    """
    Returns up to `n` random text chunks from the ingestion database.
    """
    out = get_sample_chunks(n)
    return {"chunks": out, "count": len(out)}