# app/middleware/api_key.py

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

API_KEY = os.getenv("API_KEY")
HEADER_NAME = "X-API-Key"

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path in (
            "/chat", "/ingest", "/chat-test"
        ):
            key = request.headers.get(HEADER_NAME)
            if not key or key != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API Key"},
                )
        return await call_next(request)
