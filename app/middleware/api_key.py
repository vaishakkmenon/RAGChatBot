# app/middleware/api_key.py

import os
from ..settings import settings
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

API_KEY = os.getenv("API_KEY")
HEADER_NAME = "X-API-Key"

class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        # Double-check at startup too (redundant with main.py guard, but self-contained)
        if not settings.api_key:
            raise RuntimeError(
                "Missing API_KEY environment variable. Please set API_KEY before starting the application."
            )

    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if api_key != settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing X-API-Key header"
            )
        return await call_next(request)
