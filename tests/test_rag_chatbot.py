# tests/test_rag_chatbot.py

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.ingest import chunk_text
from app.settings import settings

client = TestClient(main.app)
API_HEADERS = {"X-API-Key": settings.api_key}


@pytest.fixture(autouse=True)
def isolate_docs_dir(monkeypatch, tmp_path):
    """
    Override docs_dir so ingestion in tests accepts tmp paths.
    """
    monkeypatch.setattr(settings, "docs_dir", str(tmp_path))


# 1) POST /chat (non-streaming) → JSON ChatResponse
def test_chat_json_response(monkeypatch):
    monkeypatch.setattr(main, "search", lambda q, k, d: [
        {"id": "42", "source": "/foo/bar.md", "text": "Lorem ipsum"}
    ])
    class DummyClient:
        def chat(self, *args, **kwargs):
            return {"message": {"content": "The answer"}}
    monkeypatch.setattr(main, "_CLIENT", DummyClient())

    resp = client.post(
        "/chat",
        json={"question": "What?"},
        headers=API_HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The answer"
    srcs = body["sources"]
    assert len(srcs) == 1
    s0 = srcs[0]
    assert s0["index"] == 1
    assert s0["id"] == "42"
    assert s0["source"].endswith("bar.md")
    assert s0["text"] == "Lorem ipsum"


# 2) GET /chat/stream → SSE events
def test_chat_sse_stream(monkeypatch):
    monkeypatch.setattr(main, "search", lambda q, k, d: [])

    def fake_stream(model, messages, options, stream):
        yield {"message": {"content": ""}}        # meta
        yield {"message": {"content": "Hello "}}  # token
        yield {"message": {"content": "world!"}}  # token

    monkeypatch.setattr(main._CLIENT, "chat", fake_stream)

    with client.stream(
        "GET",
        "/chat/stream",
        params={"question": "Hi", "max_distance": 0.5},
        headers=API_HEADERS,
    ) as resp:
        assert resp.status_code == 200
        raw = resp.read()
        text = raw.decode("utf-8")

    assert "event: meta" in text
    # allow for the space after colon in "type": "token"
    assert 'data: {"type": "token"' in text
    assert "event: final" in text
    assert "event: done" in text


# 3) POST /ingest handles .txt/.md, rejects others
def test_ingest_txt_and_md(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("foo")
    (d / "b.md").write_text("bar")

    resp = client.post(
        "/ingest",
        json={"paths": [str(d)]},
        headers=API_HEADERS,
    )
    assert resp.status_code == 200
    count = resp.json()["ingested_chunks"]
    assert count == 2

def test_ingest_no_valid_files(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "x.pdf").write_text("pdf")
    resp = client.post(
        "/ingest",
        json={"paths": [str(d)]},
        headers=API_HEADERS,
    )
    assert resp.status_code == 400

def test_ingest_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    resp = client.post(
        "/ingest",
        json={"paths": [str(d)]},
        headers=API_HEADERS,
    )
    assert resp.status_code == 400


# 4) chunk_text edge-cases
@pytest.mark.parametrize("text,expected", [
    ("", []),
    ("   \n\t  ", []),
    ("small", ["small"]),
])
def test_chunk_text_edge_cases(text, expected):
    assert chunk_text(text, chunk_size=10, overlap=2) == expected


# 5) GET /debug-search respects max_distance
def test_debug_search_varying_distance(monkeypatch):
    monkeypatch.setattr(main, "search", lambda q, k, d: [
        {"id": str(i), "source": "s", "text": "t"} for i in range(int(d * 10))
    ])

    high = client.get(
        "/debug-search",
        params={"q": "hi", "k": 5, "max_distance": 1.0},
        headers=API_HEADERS,
    )
    low = client.get(
        "/debug-search",
        params={"q": "hi", "k": 5, "max_distance": 0.2},
        headers=API_HEADERS,
    )
    assert high.status_code == 200 and low.status_code == 200

    hbody = high.json()
    lbody = low.json()
    assert len(hbody["matches"]) > len(lbody["matches"])
    assert hbody["count"] == len(hbody["matches"])
    assert lbody["count"] == len(lbody["matches"])