from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi import Request

class MaxSizeMiddleware(BaseHTTPMiddleware):
    """
    Starlette/FastAPI middleware to enforce a maximum allowed HTTP POST body size.
    Returns HTTP 413 if Content-Length exceeds the configured byte limit.
    
    Args:
        app: The ASGI app to wrap.
        max_bytes (int): Maximum allowed request body size in bytes.
    """
    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        """
        Checks Content-Length on POST requests, returning 413 if over limit.
        Allows all non-POST methods to proceed without check.
        """
        if request.method == "POST":
            cl = request.headers.get("content-length")
            if cl is not None:
                try:
                    n = int(cl)
                except ValueError:
                    n = None
                if n is not None and n > self.max_bytes:
                    # IMPORTANT: return a response, don't raise here
                    return JSONResponse({"detail": "payload too large"}, status_code=413)
        return await call_next(request)
