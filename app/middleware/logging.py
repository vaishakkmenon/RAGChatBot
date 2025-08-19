import json
import logging
import os
import sys
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Logger for emitting one JSON line per request for API monitoring.
json_logger = logging.getLogger("app.json")
if not json_logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    json_logger.addHandler(h)
json_logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
json_logger.propagate = False

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware to log a single JSON line for each HTTP request.
    
    Logs:
        - request_id (UUID4)
        - method, path, status
        - elapsed_ms (wall time)
        - ollama_host, ollama_model
        - client_ip
        - content_length (from headers)
    Adds X-Request-Id to every response for traceability.
    """
    async def dispatch(self, request, call_next):
        """
        Handles incoming request, logging info at start and completion (even on exceptions).
        """
        rid = str(uuid.uuid4())
        start = time.perf_counter()

        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            json_logger.info(json.dumps({
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "elapsed_ms": elapsed_ms,
                "ollama_host": OLLAMA_HOST,
                "ollama_model": OLLAMA_MODEL,
                "client_ip": getattr(request.client, "host", None),
                "content_length": request.headers.get("content-length"),
            }, separators=(",", ":")))
            raise

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        json_logger.info(json.dumps({
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "elapsed_ms": elapsed_ms,
            "ollama_host": OLLAMA_HOST,
            "ollama_model": OLLAMA_MODEL,
            "client_ip": getattr(request.client, "host", None),
            "content_length": request.headers.get("content-length"),
        }, separators=(",", ":")))

        response.headers["X-Request-Id"] = rid
        return response
