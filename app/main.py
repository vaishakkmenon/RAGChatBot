import os, asyncio, time, ollama, socket, httpx, logging, json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .settings import settings
from .ingest import ingest_paths
from .retrieval import search, get_sample_chunks
from .middleware.api_key import APIKeyMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.max_size import MaxSizeMiddleware
from .metrics import rag_retrieval_chunks, rag_llm_request_total, rag_llm_latency_seconds
from .models import (
    QuestionRequest,
    IngestRequest,
    IngestResponse,
    ChatRequest,
    ChatResponse,
    ChatSource,
    ErrorResponse,
)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(MaxSizeMiddleware, max_bytes=MAX_BYTES)
app.add_middleware(APIKeyMiddleware)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

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
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
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
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def ingest_data(req: IngestRequest):
    """
    Ingest Markdown or plain text files from local paths (file or directory). Chunks are indexed for later retrieval.
    """
    paths = req.paths
    try:
        doc_len = ingest_paths(paths)
    except HTTPException:
        raise
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
    responses={
        200: {"content": {"application/json": {}, "text/event-stream": {}}},
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
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
    ),
    stream: bool = Query(False, description="If true, stream tokens via SSE."),
):
    """
    Retrieval-augmented question answering: finds the most relevant chunks,
    sends as context to LLM, and returns the answer with citations.

    - Default: returns a JSON ChatResponse (non-streaming).
    - If `stream=true`: streams Server-Sent Events (SSE) with tokens and a final done event.
    """
    user_question = req.question
    top_k = req.top_k if req.top_k is not None else settings.top_k
    retrieved_chunks = search(user_question, top_k, max_distance)
    rag_retrieval_chunks.observe(len(retrieved_chunks))

    numbered = []
    for i, m in enumerate(retrieved_chunks, start=1):
        numbered.append(f"[{i}] {m['source']}\n{m['text']}")
    context = "\n\n".join(numbered)

    sources_payload = [
        {
            "index": i,
            "id": m["id"],
            "source": m["source"],
            "filename": Path(m["source"]).name,
            "text": m["text"],
        }
        for i, m in enumerate(retrieved_chunks, start=1)
    ]
    
    refs = "\n".join(f"[{s['index']}] {Path(s['source']).name}" for s in sources_payload)
    llm_prompt = (
        "You are a concise assistant. Answer the question using ONLY the information below.\n"
        "Cite sources in your answer using the reference numbers like [1], [2]. "
        "List the exact filenames (not paths) from the provided sources.\n"
        "If the answer is not contained in the context, say you do not know.\n\n"
        f"References (use these numbers):\n{refs}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n\n"
        "Answer:"
    )

    # --- Streaming path (SSE) ---
    if stream:
        def sse():
            start = time.perf_counter()

            # 1) Send sources immediately so the client can render citations
            meta = {"type": "meta", "sources": sources_payload}
            yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

            try:
                # 2) Stream tokens from Ollama
                answer_parts = []
                buf = []

                for part in _CLIENT.chat(
                    model=settings.ollama_model,
                    messages=[{"role": "user", "content": llm_prompt}],
                    options={"num_ctx": settings.num_ctx},
                    stream=True,
                ):
                    chunk = part.get("message", {}).get("content", "")
                    if not chunk:
                        continue

                    # collect for final payload
                    answer_parts.append(chunk)

                    # coalesce small fragments for UI smoothness
                    buf.append(chunk)
                    if len("".join(buf)) >= 64 or "\n" in chunk:
                        out = "".join(buf); buf.clear()
                        yield f"data: {json.dumps({'type': 'token', 'content': out}, ensure_ascii=False)}\n\n"

                # flush any remainder
                if buf:
                    yield f"data: {json.dumps({'type': 'token', 'content': ''.join(buf)}, ensure_ascii=False)}\n\n"

                # 3) Emit final structured payload (ChatResponse shape)
                final_payload = {
                    "answer": "".join(answer_parts),
                    "sources": [
                        {"index": s["index"], "id": s["id"], "source": s["source"], "text": s["text"], "filename": s["filename"]}
                        for s in sources_payload
                    ],
                }
                yield f"event: final\ndata: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

                # 4) Done + metrics
                elapsed = time.perf_counter() - start
                rag_llm_latency_seconds.observe(elapsed)
                rag_llm_request_total.labels(status="ok").inc()
                yield f"event: done\ndata: {json.dumps({'type':'done','elapsed_ms': int(elapsed*1000)})}\n\n"

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
                    httpx.TransportError, OSError, socket.gaierror, socket.timeout,
                    ConnectionResetError, BrokenPipeError):
                rag_llm_request_total.labels(status="error").inc()
                yield f"event: error\ndata: {json.dumps({'type':'error','message':'Ollama unreachable'})}\n\n"
            except (httpx.HTTPStatusError, ValueError, KeyError, TypeError):
                rag_llm_request_total.labels(status="error").inc()
                yield f"event: error\ndata: {json.dumps({'type':'error','message':'Upstream error'})}\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)

    # --- Non-streaming path (unchanged JSON response) ---
    async def _call_chat():
        return await run_in_threadpool(
            _CLIENT.chat,
            model=settings.ollama_model,
            messages=[{"role": "user", "content": llm_prompt}],
            options={"num_ctx": settings.num_ctx},
        )

    start = time.perf_counter()
    try:
        resp = await asyncio.wait_for(_call_chat(), timeout=settings.ollama_timeout)
        elapsed = time.perf_counter() - start

        rag_llm_latency_seconds.observe(elapsed)
        rag_llm_request_total.labels(status="ok").inc()

        answer = resp.get("message", {}).get("content", "")
        return ChatResponse(
            answer=answer,
            sources=[
                ChatSource(index=s["index"], id=s["id"], source=s["source"], text=s["text"])
                for s in sources_payload
            ],
        )

    except asyncio.TimeoutError:
        rag_llm_request_total.labels(status="timeout").inc()
        raise HTTPException(status_code=504, detail=f"Upstream timeout after {settings.ollama_timeout}s")

    except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
            httpx.TransportError, OSError, socket.gaierror, socket.timeout,
            ConnectionResetError, BrokenPipeError) as e:
        rag_llm_request_total.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Ollama unreachable") from e

    except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as e:
        rag_llm_request_total.labels(status="error").inc()
        raise HTTPException(status_code=502, detail="Upstream error") from e

@app.get("/chat/stream")
async def chat_stream(question: str, max_distance: float = 0.65):
    return await chat(ChatRequest(question=question), max_distance=max_distance, stream=True)

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