import asyncio
import json
import logging
import os
import socket
import time
import httpx
import ollama

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
    description="A fully local Retrieval-Augmented Generation chatbot using Ollama LLM and ChromaDB, with source citation and tight OpenAPI docs.",
    version="0.2.2",
    summary="Self-hosted RAG chatbot using Ollama, SentenceTransformers, and ChromaDB."
)

# Only allow our React front-end on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key enforcement
app.add_middleware(APIKeyMiddleware)
# Request size limit for uploads
app.add_middleware(MaxSizeMiddleware, max_bytes=MAX_BYTES)
# Structured JSON logging
app.add_middleware(LoggingMiddleware)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# --- RAG helper functions: retrieval gating, de-duplication, keyword filter, and prompt building ---
import hashlib, re

SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "yo", "sup"}
CITATION_RE = re.compile(r"\s*\[\d+\]")


def should_retrieve(q: str) -> bool:
    q = (q or "").strip().lower()
    if len(q.split()) <= 3:
        return False
    if q in SMALL_TALK or any(q.startswith(w) for w in ("hi", "hello", "hey")):
        return False
    return True

def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for m in chunks:
        key = (m.get("source",""), hashlib.sha256(m.get("text","").strip().encode("utf-8")).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out

def _keyword_filter(chunks: list[dict], query: str) -> list[dict]:
    """Keep chunks that match strong query terms; if RAG terms appear in the query,
    require at least one of them in the chunk text."""
    q = (query or "").lower()
    strong_terms = {"rag", "retrieval", "augmented", "generation"}
    text_terms = {t for t in strong_terms if t in q}
    if text_terms:
        filtered = [c for c in chunks if any(t in c.get("text","").lower() for t in text_terms)]
        return filtered or chunks
    # fallback: previous lightweight filter on longer tokens (>=4 chars) but NOT common stopwords
    import re
    toks = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", q) if len(t) >= 4]
    STOP = {"this","that","with","from","your","about","into","over","under","which","have","will","been","they","their","there","these","those","where","when","what","want","need","like","just","make","take","give","work","good","more","some","such","also","very","even","than","them","then","here","only","much","many","most","each","other","into","upon","after","before","because"}
    toks = [t for t in toks if t not in STOP]
    if not toks:
        return chunks
    filtered = [c for c in chunks if sum(t in c.get("text","").lower() for t in toks) >= 2]
    return filtered or chunks

STRICT_SYS_PROMPT = (
    "EXTRACTIVE MODE. Answer ONLY by copying text spans verbatim from the Context below. "
    "Do NOT rephrase, summarize, or add any words that are not present in the Context. "
    "If the Context does not explicitly contain the answer, reply exactly: I don't know."
)

SYS_PROMPT = (
    "You are a helpful RAG assistant. If relevant sources are provided, use them to ground your answer. "
    "Prefer concise, direct answers. Do not add bracketed citation markers like [1] unless the user explicitly asks for citations. "
    "If sources are not relevant, answer briefly without fabricating citations."
)

def _format_sources(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = []
    for i, c in enumerate(chunks, start=1):
        try:
            lines.append(f"[{i}] {Path(c['source']).name}\n{c['text']}")
        except Exception:
            lines.append(f"[{i}] {c.get('source','unknown')}\n{c.get('text','')}")
    return "\n\n".join(lines)

def build_messages(question: str, chunks: list[dict], strict: bool = False) -> list[dict]:
    sys_prompt = STRICT_SYS_PROMPT if strict else SYS_PROMPT
    user = f"Question: {question.strip()}\n"
    ctx = _format_sources(chunks)
    if ctx:
        user += f"\nContext:\n{ctx}\n\nFollow the system instructions precisely."
    else:
        user += "\nNo external context provided."
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]

@app.get("/health/ollama")
async def health_ollama():
    t0 = time.time()
    ok = False
    model_ready = False
    try:
        _ = _CLIENT.list()
        ok = True

        try:
            _CLIENT.show(settings.ollama_model)
            _CLIENT.chat(
                model=settings.ollama_model,
                messages=[{"role": "user", "content": "ping"}],
                options={"num_ctx": settings.num_ctx, "num_predict": 1},
            )
            model_ready = True
        except Exception:
            model_ready = False

    except Exception:
        ok = False

    return {
        "ok": ok,
        "host": settings.ollama_host,
        "model": settings.ollama_model,
        "model_ready": model_ready,
        "num_ctx": settings.num_ctx,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest .md and .txt files into the vector store",
)
async def ingest(req: IngestRequest):
    try:
        added = await run_in_threadpool(ingest_paths, req.paths)
        return IngestResponse(ingested_chunks=added)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))

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
        0.45,
        ge=0.0, le=1.0,
        description=(
            "Maximum vector distance allowed for retrieved chunks. "
            "Lower = closer match, higher = looser match. "
            "Typical values: 0.35–0.50 for cosine distance."
        )
    ),
    stream: bool = Query(False, description="If true, stream tokens via SSE."),
    grounded_only: bool = Query(False, description="If true, use only provided sources; otherwise say 'I don't know.'"),
    temperature: float = Query(0.2, ge=0.0, le=1.5, description="Sampling temperature for the model"),
):
    """
    Retrieval-augmented question answering: finds the most relevant chunks,
    sends as context to LLM, and returns the answer with citations.

    - Default: returns a JSON ChatResponse (non-streaming).
    - If `stream=true`: streams Server-Sent Events (SSE) with tokens and a final done event.
    """
    user_question = req.question
    top_k = req.top_k if req.top_k is not None else settings.top_k

    do_retrieve = should_retrieve(user_question)
    retrieved_chunks = search(user_question, top_k, max_distance) if do_retrieve else []
    retrieved_chunks = _dedupe_chunks(retrieved_chunks)
    retrieved_chunks = _keyword_filter(retrieved_chunks, user_question)
    
    if grounded_only and not retrieved_chunks:
        return ChatResponse(answer="I don't know.", sources=[])
    
    rag_retrieval_chunks.observe(len(retrieved_chunks))

    sources_payload = [
        {
            "index": i,
            "id": m["id"],
            "source": m["source"],
            "filename": Path(m["source"]).name if isinstance(m.get("source"), str) else "unknown",
            "text": m["text"],
        }
        for i, m in enumerate(retrieved_chunks, start=1)
    ]

    messages = build_messages(user_question, retrieved_chunks, strict=grounded_only)

    # --- Streaming path (SSE) ---
    if stream:
        def sse():
            start = time.perf_counter()

            # 1) Send sources immediately so the client can render citations
            meta = {"type": "meta", "sources": sources_payload}
            yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

            try:
                # 2) Stream tokens
                buf = []
                answer_parts = []
                for part in _CLIENT.chat(
                    model=settings.ollama_model,
                    messages=messages,
                    options={"num_ctx": settings.num_ctx, "temperature": temperature, "top_p": 0},
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
                        out = "".join(buf)
                        buf.clear()
                        yield f"event: token\ndata: {{\"type\": \"token\", \"content\": {json.dumps(out, ensure_ascii=False)} }}\n\n"

                # 3) flush tail
                if buf:
                    out = "".join(buf)
                    buf.clear()
                    yield f"event: token\ndata: {{\"type\": \"token\", \"content\": {json.dumps(out, ensure_ascii=False)} }}\n\n"

                # 4) Emit final structured payload (ChatResponse shape)
                final_answer = CITATION_RE.sub("", "".join(answer_parts))
                final_payload = {
                    "answer": final_answer,
                    "sources": [
                        {"index": s["index"], "id": s["id"], "source": s["source"], "text": s["text"], "filename": s["filename"]}
                        for s in sources_payload
                    ],
                }
                yield f"event: final\ndata: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

                # 5) Done + metrics
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

    # --- Non-streaming path ---
    try:
        t0 = time.perf_counter()
        async def _call_chat():
            return await run_in_threadpool(
                _CLIENT.chat,
                model=settings.ollama_model,
                messages=messages,
                options={"num_ctx": settings.num_ctx, "temperature": temperature, "top_p": 0},
            )

        resp = await asyncio.wait_for(_call_chat(), timeout=REQUEST_TIMEOUT_S)
        elapsed = time.perf_counter() - t0
        rag_llm_latency_seconds.observe(elapsed)
        rag_llm_request_total.labels(status="ok").inc()

        answer_text = (resp.get("message", {}) or {}).get("content", "").strip()
        answer_text = CITATION_RE.sub("", answer_text)
        return ChatResponse(
            answer=answer_text,
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
        raise HTTPException(status_code=503, detail="Ollama unreachable")

    except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as e:
        rag_llm_request_total.labels(status="error").inc()
        raise HTTPException(status_code=502, detail="Upstream error")

# --- Debug route to preview retrieval (non-LLM) ---
@app.get("/debug/search")
async def debug_search(q: str, k: int = 5, max_distance: float = 0.45):
    return search(q, k, max_distance)

# --- Sample chunks (for UI demos) ---
@app.get("/debug/samples")
async def debug_samples(n: int = 4):
    return get_sample_chunks(n)