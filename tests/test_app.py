# tests/test_app.py
import io
import json
import pytest
import asyncio
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.main as main
from app.main import _CLIENT
from app.settings import settings
from app.ingest import chunk_text
from app.middleware.logging import json_logger
from app.middleware.max_size import MaxSizeMiddleware

client = TestClient(main.app)

API_HEADERS = {"X-API-Key": settings.api_key}

# --------------------------------------------------------------------
# Fixture to isolate settings.docs_dir
# --------------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolate_docs_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "docs_dir", str(tmp_path))
    yield


# --------------------------------------------------------------------
# Helper for LLM responses
# --------------------------------------------------------------------
class DummyResp:
    def __init__(self, content):
        self._content = content

    def get(self, key, default=None):
        if key == "message":
            return {"content": self._content}
        return default


# --------------------------------------------------------------------
# 1) Robust /chat tests (happy + error + metrics)
# --------------------------------------------------------------------
def test_chat_happy_path(monkeypatch):
    monkeypatch.setattr(main, "search", lambda q, k, d: [{"id":"1","source":"s","text":"t"}])
    monkeypatch.setattr(_CLIENT, "chat", lambda **kw: DummyResp("ok"))
    monkeypatch.setattr(main.rag_retrieval_chunks, "observe", lambda v: None)
    monkeypatch.setattr(main.rag_llm_request_total, "labels",
                        lambda **kw: type("", (), {"inc": lambda self: None})())
    monkeypatch.setattr(main.rag_llm_latency_seconds, "observe", lambda v: None)

    resp = client.post(
        "/chat",
        json={"question": "q"},
        headers=API_HEADERS,
    )
    assert resp.status_code == 200

@pytest.mark.parametrize("exc,code", [
    (asyncio.TimeoutError(), 504),
    (HTTPException(status_code=503, detail="boom"), 503),
])
def test_chat_error_paths(monkeypatch, exc, code):
    monkeypatch.setattr(main, "search", lambda *a, **k: [])
    monkeypatch.setattr(main.asyncio, "wait_for", lambda coro, timeout: (_ for _ in ()).throw(exc))
    resp = client.post(
        "/chat",
        json={"question": "q"},
        headers=API_HEADERS,
    )
    assert resp.status_code == code

def test_chat_validation_error():
    resp = client.post(
        "/chat",
        json={},
        headers=API_HEADERS,
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------
# 2) /chat-test and /health/ollama
# --------------------------------------------------------------------
def test_chat_test_happy(monkeypatch):
    monkeypatch.setattr(_CLIENT, "chat", lambda **kw: DummyResp("test"))
    resp = client.post(
        "/chat-test",
        json={"question": "hey"},
        headers=API_HEADERS,
    )
    assert resp.status_code == 200

def test_chat_test_validation_error():
    resp = client.post(
        "/chat-test",
        json={},
        headers=API_HEADERS,
    )
    assert resp.status_code == 422

def test_health_ollama_happy(monkeypatch):
    monkeypatch.setattr(_CLIENT, "list", lambda: {"models":[{"name":settings.ollama_model}]})
    resp = client.get(
        "/health/ollama",
        headers=API_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_ready"] is True


# --------------------------------------------------------------------
# 3) /debug-search boundaries & errors
# --------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 20])
def test_debug_search_k_boundaries(monkeypatch, k):
    captured = {}
    def fake_search(q, k_, d):
        captured.update({"query": q, "k": k_, "max_distance": d})
        # return dummy chunks with the required keys
        return [{"id": str(i), "source": f"s{i}", "text": f"t{i}"} for i in range(k_)]
    monkeypatch.setattr(main, "search", fake_search)

    resp = client.get(
        "/debug-search", 
        params={"q": "x", "k": k, "max_distance": 0.5},
        headers=API_HEADERS,
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["count"] == k
    # now it’s safe to inspect the returned matches
    assert all(m["id"] == str(i) for i, m in enumerate(body["matches"]))


@pytest.mark.parametrize("param,val", [
    ("k", 0), ("k", 21),
    ("max_distance", -0.1), ("max_distance", 2.1),
])
def test_debug_search_invalid_params(param, val):
    params = {"q": "x", "k": 4, "max_distance": 0.5}
    params[param] = val
    resp = client.get(
        "/debug-search", 
        params=params,
        headers=API_HEADERS,
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------
# 4) /debug-ingest endpoint
# --------------------------------------------------------------------
def test_debug_ingest(monkeypatch):
    # prepare some chunks
    monkeypatch.setattr(main, "get_sample_chunks", lambda n: [f"c{i}" for i in range(n)])
    resp = client.get(
        "/debug-ingest", 
        params={"n":5},
        headers=API_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 5
    assert data["chunks"] == ["c0","c1","c2","c3","c4"]


# --------------------------------------------------------------------
# 5) /metrics exposition
# --------------------------------------------------------------------
def test_metrics_exposition():
    resp = client.get(
        "/metrics", 
        headers=API_HEADERS,
    )
    assert resp.status_code == 200
    text = resp.text
    assert "rag_retrieval_chunks" in text
    assert "rag_llm_request_total" in text
    assert "rag_llm_latency_seconds" in text


# --------------------------------------------------------------------
# 6) MaxSizeMiddleware
# --------------------------------------------------------------------
def test_max_size_middleware(tmp_path, monkeypatch):
    # create a small app with the middleware
    from fastapi import FastAPI
    from fastapi.testclient import TestClient as TC
    small = FastAPI()
    small.add_middleware(MaxSizeMiddleware, max_bytes=10)
    @small.post("/echo")
    def echo(body: dict):
        return body
    tc = TC(small)
    # under limit
    r1 = tc.post("/echo", json={"a":"1"})
    assert r1.status_code == 200
    # over limit (>10 bytes)
    payload = {"x": "y"*20}
    r2 = tc.post("/echo", json=payload)
    assert r2.status_code == 413


# --------------------------------------------------------------------
# 7) LoggingMiddleware
# --------------------------------------------------------------------
def test_logging_middleware_logs(monkeypatch, capsys):
    # capture json_logger output
    # swap in our own StreamHandler on the JSON logger
    from logging import StreamHandler
    stream = io.StringIO()
    handler = StreamHandler(stream)
    monkeypatch.setattr(json_logger, "handlers", [handler])

    resp = client.get(
        "/", 
        headers=API_HEADERS,
    )
    assert resp.status_code == 200

    # fetch the single JSON log line
    log_line = stream.getvalue().strip()
    data = json.loads(log_line)
    assert "request_id" in data and "method" in data and "path" in data and "status" in data

# --------------------------------------------------------------------
# 8) /ingest error & nested directory + file ingestion
# --------------------------------------------------------------------
def test_ingest_not_found():
    resp = client.post(
        "/ingest", 
        json={"paths":["/no/such"]},
        headers=API_HEADERS,
    )
    assert resp.status_code == 400

def test_ingest_permission(monkeypatch):
    monkeypatch.setattr(main, "ingest_paths",
                        lambda paths: (_ for _ in ()).throw(PermissionError()))
    resp = client.post(
        "/ingest", 
        json={"paths":["any"]},
        headers=API_HEADERS,
    )
    assert resp.status_code == 403

def test_ingest_generic_error(monkeypatch):
    monkeypatch.setattr(main, "ingest_paths",
                        lambda paths: (_ for _ in ()).throw(ValueError()))
    resp = client.post(
        "/ingest", 
        json={"paths":["any"]},
        headers=API_HEADERS,
    )
    assert resp.status_code == 500

def test_ingest_nested_and_file(tmp_path):
    # create nested dirs with mixed files
    base = tmp_path / "docs"
    sub = base / "sub"
    sub.mkdir(parents=True)
    (base / "a.md").write_text("A")
    (sub / "b.txt").write_text("B")

    # ingest by directory
    r1 = client.post(
        "/ingest",
        json={"paths": [str(base)]},
        headers=API_HEADERS,
    )
    assert r1.status_code == 200
    assert r1.json()["ingested_chunks"] == 2

    # ingest by file
    r2 = client.post(
        "/ingest",
        json={"paths": [str(sub / "b.txt")]},
        headers=API_HEADERS,
    )
    assert r2.status_code == 200
    assert r2.json()["ingested_chunks"] == 1


# --------------------------------------------------------------------
# 9) chunk_text edge & large text & overlap=0
# --------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("", []),
    ("  \n ", []),
    ("exact", ["exact"]),
])
def test_chunk_empty_and_exact(text, expected):
    assert chunk_text(text, chunk_size=5, overlap=1) == expected

def test_chunk_zero_overlap_and_large_text():
    text1 = "abcdefghij"
    chunks1 = chunk_text(text1, chunk_size=4, overlap=0)
    assert "".join(chunks1) == text1
    assert all(c for c in chunks1)

    text2 = "x" * 100
    chunks2 = chunk_text(text2, chunk_size=10, overlap=2)
    reconstructed = "".join(
        chunks2[0] if i == 0 else chunks2[i][2:]
        for i in range(len(chunks2))
    )
    assert reconstructed == text2
    assert len(chunks2) >= 1