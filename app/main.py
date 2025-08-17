import os, asyncio, time
import ollama
import socket
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from .retrieval import search
from .settings import settings
from .ingest import ingest_paths
from .middleware.logging import LoggingMiddleware
from .middleware.max_size import MaxSizeMiddleware
from .models import QuestionRequest, IngestRequest, IngestResponse, ChatRequest



import logging
logger = logging.getLogger(__name__)

_CLIENT = ollama.Client(host=settings.ollama_host)
_MODEL = settings.ollama_model
_NUM_CTX = settings.num_ctx
REQUEST_TIMEOUT_S = settings.ollama_timeout
MAX_BYTES = settings.max_bytes

app = FastAPI(title="RAGChatBot")
app.add_middleware(LoggingMiddleware)
app.add_middleware(MaxSizeMiddleware, max_bytes=MAX_BYTES)

@app.get("/")
def root():
    return {"ok": True, "message": "Hello from RAGChatBot"}

@app.post("/chat-test")
async def chat_test(req: QuestionRequest):
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
        
@app.get("/health/ollama")
async def ollama_health():
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

@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(req: IngestRequest):
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

@app.get("/debug-search")
def debug_search(
    q: str = Query(..., description="Search query string", min_length=1),
    k: int = Query(4, description="Number of top results to return", ge=1, le=20),
    similarity: float = Query(0.35, ge=0.0, le=1.0, description="Minimum similarity threshold (lower = stricter, 0.25-0.4 typical)")
):
    results = search(q, k, similarity)
    logger.info(f"Debug-search: q='{q}', k={k}, similarity={similarity}, matches={len(results)}")
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

@app.post("/chat")
async def chat(
    req: ChatRequest, 
    similarity: float = Query(0.35, ge=0.0, le=1.0, description="Minimum similarity threshold (lower = stricter, 0.25-0.4 typical)")
):
    user_question = req.question
    top_k = req.top_k if req.top_k is not None else settings.top_k
    retrieved_chunks = search(user_question, top_k, similarity)
    context_chunks = [match['text'] for match in retrieved_chunks]
    context = "\n\n".join(context_chunks)
    llm_prompt = (
        "Answer the question using only the information below.\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {user_question}"
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
        return {
            "answer": answer,
            "sources": [
                {"id": match["id"], "source": match["source"]}
                for match in retrieved_chunks
            ]
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Upstream timeout after {settings.ollama_timeout}s")
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
            httpx.TransportError, OSError, socket.gaierror, socket.timeout,
            ConnectionResetError, BrokenPipeError) as e:
        raise HTTPException(status_code=503, detail="Ollama unreachable") from e
    except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=502, detail="Upstream error") from e
