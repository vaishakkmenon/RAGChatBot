import os, asyncio, time
import ollama
import socket
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from .settings import settings
from .models import QuestionRequest
from .middleware.max_size import MaxSizeMiddleware
from .middleware.logging import LoggingMiddleware


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