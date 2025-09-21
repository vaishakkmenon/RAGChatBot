# main.py — FastAPI RAG Chatbot (cleaned & organized)
# - Keeps endpoints: /health, /ingest, /rc, /chat (+ debug routes)
# - Keeps middleware, metrics, CORS
# - Keeps A2 deterministic evidence span, A3 reranker, extractive/generative modes, abstention gates
# - Removes dead code & duplicates; merges duplicate constants; consistent naming/docstrings

import asyncio
import json
import logging
import os
import socket
import time
import re
from typing import Optional, Tuple, Sequence, Dict, Any, List

import ollama
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .settings import settings
from .ingest import ingest_paths
from .retrieval import search, get_sample_chunks
from .middleware.api_key import APIKeyMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.max_size import MaxSizeMiddleware
from .metrics import (
    rag_retrieval_chunks,
    rag_llm_request_total,
    rag_llm_latency_seconds,
)
from .models import (
    IngestRequest,
    IngestResponse,
    ChatRequest,
    ChatResponse,
    ChatSource,
    RCRequest,
    EvidenceSpan,
)

# ------------------------------------------------------------------------------
# Logging & Globals
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Ollama client & core settings
_CLIENT = ollama.Client(host=settings.ollama_host)
_MODEL = settings.ollama_model
_NUM_CTX = settings.num_ctx
REQUEST_TIMEOUT_S = settings.ollama_timeout
MAX_BYTES = settings.max_bytes

# RC behavior toggles (env)
RC_WINDOW_PAD = int(os.getenv("RC_WINDOW_PAD", "48"))             # Local refine window
RC_ALWAYS_SHRINK = os.getenv("RC_ALWAYS_SHRINK", "true").lower() == "true"
RC_USE_TYPE_HINTS = os.getenv("RC_USE_TYPE_HINTS", "false").lower() == "true"

# Sentence splitter (merged duplicate)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# ------------------------------------------------------------------------------
# System prompts (small, stable)
# ------------------------------------------------------------------------------
RC_JSON_SYS_PROMPT = (
    "You are an EXTRACTIVE QA tool. Given a question and a context string, "
    "return the SHORTEST contiguous substring of the context that answers the question. "
    "If the context does not explicitly contain the answer, mark it unanswerable.\n"
    "Respond with JSON ONLY (no prose) in exactly this shape:\n"
    '{ "answerable": true|false, "start": <int>, "end": <int> }\n'
    "- Indices are 0-based character offsets into the EXACT context string.\n"
    "- If answerable=false, omit start/end or set both to -1.\n"
    "- Among valid answers, choose the one with the FEWEST characters.\n"
)

REFINE_SYS = (
    "You are a STRICT extractor. Given a question and ONLY the provided text, "
    "return the SHORTEST answer span indices within the text.\n"
    'Respond with JSON ONLY: {"start": int, "end": int} (0-based, within the given text). '
    "If NO answer exists within the given text, set start=-1 and end=-1."
)

CHAT_SYS_PROMPT = (
    "Answer with the SHORTEST phrase copied verbatim from the provided context. "
    "Do NOT add citations, brackets, or commentary. "
    "If the answer is not explicitly in the context, reply exactly: I don't know."
)

# ------------------------------------------------------------------------------
# FastAPI app & middleware wiring
# ------------------------------------------------------------------------------
app = FastAPI(
    title="RAGChatBot (Local $0)",
    description="Self-hosted RAG chatbot using Ollama, SentenceTransformers, and ChromaDB.",
    version="0.3.0",
    summary="Local RAG + SQuAD-v2-style RC.",
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key, request-size limit, structured logging
app.add_middleware(APIKeyMiddleware)
app.add_middleware(MaxSizeMiddleware, max_bytes=MAX_BYTES)
app.add_middleware(LoggingMiddleware)

# Prometheus /metrics
Instrumentator().instrument(app).expose(app)

# ==============================================================================
# Helpers — Span snapping & refinement (A2 core)
# ==============================================================================

def _contains_word(s: str) -> bool:
    """True if the string has at least one alphanumeric character."""
    return any(ch.isalnum() for ch in (s or ""))


def _snap_to_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Clamp to [0, len], expand if mid-token, trim whitespace and obvious quotes,
    trim trailing punctuation (keep '.' if likely part of acronym).
    """
    n = len(text)
    start = max(0, min(start, n))
    end = max(0, min(end, n))
    if end < start:
        start, end = end, start

    # Expand if cutting through words
    while start > 0 and start < n and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while end < n and end > 0 and text[end - 1].isalnum() and text[end].isalnum():
        end += 1

    # Trim whitespace
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    # Trim simple leading quotes/brackets
    while start < end and text[start] in "\"'“”‘’([{":
        start += 1

    # Trim trailing punctuation (common marks)
    while end > start and text[end - 1] in ",;:!?":
        end -= 1

    # Handle trailing '.' (keep only if acronym-like)
    if end > start and text[end - 1] == '.':
        span_wo_last = text[start:end - 1]
        if '.' not in span_wo_last:
            end -= 1

    return start, end


def _refine_span(question: str, text: str) -> tuple[int, int]:
    """
    Ask the model for a start/end span inside `text`.
    Returns (-1, -1) if unanswerable or if parsing fails.
    """
    messages = [
        {"role": "system", "content": REFINE_SYS},
        {"role": "user", "content": f"Question:\n{question}\n\nText:\n{text}\n\nReturn JSON ONLY."},
    ]
    resp = _CLIENT.chat(
        model=_MODEL,
        messages=messages,
        format="json",
        options={"num_ctx": _NUM_CTX, "temperature": 0.0, "top_p": 0},
    )
    raw = (resp.get("message", {}) or {}).get("content", "").strip()
    try:
        d = json.loads(raw)
        s = int(d.get("start", -1))
        e = int(d.get("end", -1))
        if 0 <= s < e <= len(text):
            return s, e
    except Exception:
        pass
    return -1, -1


def _refine_text(question: str, text: str) -> str:
    """
    Ask the model for the SHORTEST answer TEXT (not indices) within `text`.
    Returns "" or "I don't know." if none.
    """
    TEXT_SYS = (
        "You are a STRICT extractor. Given a question and ONLY the provided text, "
        "return the SHORTEST contiguous substring that answers the question. "
        "If none exists, reply exactly: I don't know.\n"
        'Respond with JSON ONLY as {"text": "<answer or I don\'t know>"}'
    )
    messages = [
        {"role": "system", "content": TEXT_SYS},
        {"role": "user", "content": f"Question:\n{question}\n\nText:\n{text}\n\nReturn JSON ONLY."},
    ]
    resp = _CLIENT.chat(
        model=_MODEL,
        messages=messages,
        format="json",
        options={"num_ctx": _NUM_CTX, "temperature": 0.0, "top_p": 0},
    )
    raw = (resp.get("message", {}) or {}).get("content", "").strip()
    try:
        ans = (json.loads(raw).get("text") or "").strip()
    except Exception:
        m = re.search(r'\{\s*"text"\s*:\s*"(.*?)"\s*\}', raw, re.S)
        ans = (m.group(1).strip() if m else "")
    return ans


def _find_substring_ci(hay: str, needle: str) -> tuple[int, int] | tuple[None, None]:
    """Case-insensitive substring find; returns [start,end) if found, else (None, None)."""
    if not hay or not needle:
        return None, None
    ihay = hay.lower()
    ineedle = needle.lower()
    k = ihay.find(ineedle)
    if k == -1:
        return None, None
    return k, k + len(needle)


# --- Deterministic A2 span locator (exact/CI/subsequence fallbacks) ---
_WS_RE = re.compile(r"\s+")
_TOK_RE = re.compile(r"\w+", re.UNICODE)

def _normalize_ws(s: str) -> str:
    """Collapse all whitespace to single spaces and trim."""
    return _WS_RE.sub(" ", s).strip()

def _find_span_basic(answer: str, text: str) -> Optional[Tuple[int, int]]:
    """Try exact, then case-insensitive substring match. Return (start,end) or None."""
    if not answer:
        return None
    idx = text.find(answer)
    if idx != -1:
        return idx, idx + len(answer)
    idx_ci = text.casefold().find(answer.casefold())
    if idx_ci != -1:
        return idx_ci, idx_ci + len(answer)
    return None

def _find_span_subseq(answer: str, text: str) -> Optional[Tuple[int, int]]:
    """
    Ordered token-subsequence fallback.
    Tokenize with \w+; match tokens in order (case-insensitive).
    Return a single contiguous char span from first to last token match.
    """
    ans_tokens = _TOK_RE.findall(answer)
    if not ans_tokens:
        return None
    ans_tokens_ci = [t.casefold() for t in ans_tokens]

    matches: List[Tuple[int, int]] = []
    ai = 0
    for m in _TOK_RE.finditer(text):
        if ai >= len(ans_tokens_ci):
            break
        if m.group(0).casefold() == ans_tokens_ci[ai]:
            matches.append((m.start(), m.end()))
            ai += 1
            if ai == len(ans_tokens_ci):
                break
    if ai != len(ans_tokens_ci) or not matches:
        return None
    return matches[0][0], matches[-1][1]

def locate_best_span(answer: str, sources: Sequence[ChatSource | Dict[str, Any]]) -> Optional[EvidenceSpan]:
    """
    Iterate sources in order; try basic match then subsequence. On hit, return EvidenceSpan.
    Accepts either ChatSource objects or dicts with id/text.
    """
    if not answer:
        return None
    for src in sources:
        src_id = src.id if isinstance(src, ChatSource) else str(src.get("id", ""))
        src_text = src.text if isinstance(src, ChatSource) else str(src.get("text", ""))
        if not src_text:
            continue

        hit = _find_span_basic(answer, src_text)
        if hit is None:
            norm_answer = _normalize_ws(answer)
            if norm_answer != answer:
                hit = _find_span_basic(norm_answer, src_text)
        if hit is None:
            hit = _find_span_subseq(answer, src_text)

        if hit is not None:
            s, e = hit
            if 0 <= s <= e <= len(src_text):
                return EvidenceSpan(doc_id=src_id, start=s, end=e, text=src_text[s:e])
    return None

# ==============================================================================
# Helpers — Optional type heuristics
# ==============================================================================

_PERSON_RE = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}$")

def _infer_answer_type(question: str) -> str | None:
    """Very small heuristic: 'who' → PERSON."""
    q = (question or "").strip().lower()
    if q.startswith("who") or " who " in q:
        return "PERSON"
    return None

def _is_person_like(s: str) -> bool:
    """Loose check for proper-name-like spans."""
    s = (s or "").strip()
    return bool(_PERSON_RE.match(s))

def _person_from_window(text: str, left: int, right: int) -> tuple[int | None, int | None]:
    """Search a small window near the span for a capitalized name; prefer nearest to center."""
    n = len(text)
    L = max(0, left)
    R = min(n, right)
    window = text[L:R]

    m = re.search(r"\bby ([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b", window)
    if m:
        s, e = m.span(1)
        return L + s, L + e

    center = (L + R) // 2
    best = None
    for m in re.finditer(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b", window):
        s, e = m.span(1)
        candL, candR = L + s, L + e
        dist = abs((candL + candR) // 2 - center)
        if best is None or dist < best[0]:
            best = (dist, candL, candR)
    if best:
        _, s, e = best
        return s, e
    return None, None

# ==============================================================================
# Helpers — Overlap scoring & A3 reranker + sentence/window support
# ==============================================================================

# Tiny stopword set + tokenization for lexical overlap
_STOPWORDS = {
    "the","a","an","of","to","in","on","at","for","and","or","if","is","are","was","were",
    "by","with","from","as","that","this","these","those","it","its","be","been","being",
    "which","who","whom","what","when","where","why","how"
}
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokset(s: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(s or "") if w.lower() not in _STOPWORDS}

def _overlap_score(question: str, ctx: str) -> float:
    q = _tokset(question)
    c = _tokset(ctx)
    if not q or not c:
        return 0.0
    return len(q & c) / max(1, len(q))

def _inv_distance(d: float | None) -> float:
    """Map 0..1 distance → 1..0 'closeness' with clamping."""
    if d is None:
        return 0.0
    try:
        d = float(d)
    except Exception:
        return 0.0
    d = max(0.0, min(1.0, d))
    return 1.0 - d

def rerank_chunks(question: str, chunks: Sequence[Dict[str, Any] | Any], w_lex: float) -> list:
    """
    Score = w_lex * lexical_overlap + (1 - w_lex) * inverse_distance.
    Returns a NEW list, sorted descending by score. Does not mutate input.
    Accepts dicts or objects with .text/.distance.
    """
    w = max(0.0, min(1.0, float(w_lex)))
    def score(c) -> float:
        text = _gx(c, "text", "") or ""
        d = _gx(c, "distance", 1.0)
        lex = _overlap_score(question, text)
        invd = _inv_distance(d)
        return w * lex + (1.0 - w) * invd
    return sorted(list(chunks), key=score, reverse=True)

def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Return [L, R) of the sentence that contains the span [start, end).
    Falls back to the whole text if boundaries can't be found.
    """
    sentences = _SENT_SPLIT.split(text or "")
    if not sentences:
        return 0, len(text or "")
    off = 0
    for s in sentences:
        L, R = off, off + len(s)
        if start >= L and end <= R:
            return L, R
        off = R + 1  # account for the split whitespace
    return 0, len(text or "")

# ==============================================================================
# Helpers — Misc
# ==============================================================================

def _gx(obj, key: str, default=None):
    """Generic safe getter for dicts or Pydantic-like objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    val = getattr(obj, key, None)
    return val if val is not None else default

# ==============================================================================
# Routes
# ==============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _MODEL,
        "ollama_host": settings.ollama_host,
        "socket": socket.gethostname(),
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    paths = req.paths or [settings.docs_dir]
    added = ingest_paths(paths)
    return IngestResponse(ingested_chunks=added)

# --- Reading Comprehension (SQuAD-v2-style) ---
@app.post(
    "/rc",
    response_model=ChatResponse,
    summary="Reading comprehension over provided context (extractive, minimal)",
)
async def rc(req: RCRequest, temperature: float = 0.0):
    ctx = req.context
    n = len(ctx)

    # Timeout + metrics around the LLM call
    t0 = time.time()
    try:
        primary = await asyncio.wait_for(
            run_in_threadpool(
                _CLIENT.chat,
                model=_MODEL,
                messages=[
                    {"role": "system", "content": RC_JSON_SYS_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{req.question}\n\n"
                            f"Context (index in THIS context):\n{ctx}\n\n"
                            f"Return JSON ONLY."
                        ),
                    },
                ],
                format="json",
                options={"num_ctx": _NUM_CTX, "temperature": temperature, "top_p": 0},
            ),
            timeout=REQUEST_TIMEOUT_S,
        )
        rag_llm_request_total.labels(status="ok", model=_MODEL).inc()
        rag_llm_latency_seconds.labels(status="ok", model=_MODEL).observe(time.time() - t0)
    except asyncio.TimeoutError:
        rag_llm_request_total.labels(status="timeout", model=_MODEL).inc()
        rag_llm_latency_seconds.labels(status="timeout", model=_MODEL).observe(time.time() - t0)
        raise HTTPException(status_code=504, detail="LLM timeout")
    except Exception as e:
        rag_llm_request_total.labels(status="error", model=_MODEL).inc()
        rag_llm_latency_seconds.labels(status="error", model=_MODEL).observe(time.time() - t0)
        logger.exception("LLM /rc failed")
        raise HTTPException(status_code=500, detail=str(e))

    raw = (primary.get("message", {}) or {}).get("content", "").strip()
    logger.info(f"[RC] raw: {raw}")

    try:
        d = json.loads(raw)
    except Exception:
        d = {}

    if not d or d.get("answerable") is not True:
        logger.info("[RC] primary says unanswerable → I don't know.")
        return ChatResponse(
            answer="I don't know.",
            sources=[ChatSource(index=1, id="rc_json::0", source="squad_context", text=ctx)],
        )

    raw_start = int(d.get("start", -1))
    raw_end = int(d.get("end", -1))
    start = max(0, min(raw_start, n))
    end = max(0, min(raw_end, n))
    span = ctx[start:end]
    oob = (raw_start < 0 or raw_end < 0 or raw_start >= raw_end or raw_end > n)
    logger.info(f"[RC] primary indices: start={start}, end={end}, span='{span}', oob={oob}")

    # Refine WITHIN if long
    too_long = (len(span) > 24) or (len(span.split()) > 4)
    if too_long:
        s2, e2 = _refine_span(req.question, span)
        logger.info(f"[RC] refine-within returned: s2={s2}, e2={e2}")
        if 0 <= s2 < e2 <= len(span):
            start = start + s2
            end = start + (e2 - s2)
            span = ctx[start:end]
            logger.info(f"[RC] after refine-within: start={start}, end={end}, span='{span}'")

    # Refine AROUND if junk
    junk = (not span) or (not _contains_word(span))
    if junk:
        left = max(0, min(start, end) - RC_WINDOW_PAD)
        right = min(n, max(start, end) + RC_WINDOW_PAD)
        window = ctx[left:right]
        s3, e3 = _refine_span(req.question, window)
        logger.info(f"[RC] refine-around window=[{left},{right}) len={len(window)} → s3={s3}, e3={e3}")
        if 0 <= s3 < e3 <= len(window):
            start = left + s3
            end = left + e3
            span = ctx[start:end]
            logger.info(f"[RC] after refine-around: start={start}, end={end}, span='{span}'")

    # Always shrink within local window (general, safe)
    if RC_ALWAYS_SHRINK:
        left = max(0, start - RC_WINDOW_PAD)
        right = min(n, end + RC_WINDOW_PAD)
        window = ctx[left:right]
        s5, e5 = _refine_span(req.question, window)
        logger.info(f"[RC] shrink-window [{left},{right}) -> s5={s5}, e5={e5}")
        if 0 <= s5 < e5 <= len(window):
            start = left + s5
            end = left + e5
            span = ctx[start:end]
            logger.info(f"[RC] after shrink-window: start={start}, end={end}, span='{span}'")

    # Snap to word boundaries and optionally fall back to text refinement
    start, end = _snap_to_word_boundaries(ctx, start, end)
    span = ctx[start:end].strip()
    logger.info(f"[RC] after snap: start={start}, end={end}, span='{span}'")

    still_long = (len(span) > 24) or (len(span.split()) > 4)
    if (not span or not _contains_word(span)) or oob or still_long:
        L = max(0, start - RC_WINDOW_PAD) if end > start else 0
        R = min(n, end + RC_WINDOW_PAD) if end > start else n
        window = ctx[L:R] if (R > L) else ctx
        ans_text = _refine_text(req.question, window)
        logger.info(f"[RC] text-fallback → '{ans_text}'")
        if ans_text and ans_text.lower() not in ("i don't know.", "i don't know"):
            s_loc, e_loc = _find_substring_ci(window, ans_text)
            if s_loc is not None and e_loc is not None:
                start = L + s_loc
                end = L + e_loc
                start, end = _snap_to_word_boundaries(ctx, start, end)
                span = ctx[start:end].strip()
                logger.info(f"[RC] after text-fallback locate: start={start}, end={end}, span='{span}'")

    # Optional type-aware correction for WHO questions
    if RC_USE_TYPE_HINTS:
        atype = _infer_answer_type(req.question)
        if atype == "PERSON" and not _is_person_like(span):
            L = max(0, start - 64)
            R = min(n, end + 64)
            ps, pe = _person_from_window(ctx, L, R)
            logger.info(f"[RC] type-hint PERSON search [{L},{R}) → ps={ps}, pe={pe}")
            if ps is not None and pe is not None:
                start, end = _snap_to_word_boundaries(ctx, ps, pe)
                span = ctx[start:end].strip()
                logger.info(f"[RC] after PERSON snap: start={start}, end={end}, span='{span}'")

    answer = span if (span and _contains_word(span)) else "I don't know."
    logger.info(f"[RC] FINAL: answer='{answer}' (start={start}, end={end})")

    return ChatResponse(
        answer=answer,
        sources=[ChatSource(index=1, id="rc_json::0", source="squad_context", text=ctx)],
    )

# --- RAG chat with distance-based abstention ---
@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    # Abstention / retrieval knobs
    grounded_only: bool = Query(True, description="Abstain when retrieval is weak."),
    null_threshold: float = Query(settings.null_threshold, ge=0.0, le=1.0),
    temperature: float = Query(0.0, ge=0.0, le=1.0),
    max_distance: float = Query(settings.max_distance, ge=0.0, le=1.0),

    # Extractive knobs
    extractive: bool = Query(False, description="Use strict extractive head (copy verbatim)."),
    alpha: float | None = Query(None, ge=0.0, le=1.0, description="Stricter extractive cutoff."),
    alpha_hits: int = Query(1, ge=0, description="Min chunks under alpha."),
    support_min: float = Query(0.15, ge=0.0, le=1.0, description="Min token-overlap for support."),
    support_window: int = Query(96, ge=16, le=512, description="Half-window (chars) around evidence."),
    span_max_distance: float | None = Query(None, ge=0.0, le=1.0, description="Max distance for the evidence chunk."),

    # A3 reranker
    rerank: bool = Query(False, description="Re-rank retrieved chunks by lexical overlap + inverse distance."),
    rerank_lex_w: float = Query(0.50, ge=0.0, le=1.0, description="Weight for lexical overlap (0..1)."),
):
    # --- 1) Retrieve ---
    k = max(1, min(20, (req.top_k or settings.top_k)))
    chunks = search(req.question, k=k, max_distance=max_distance)
    rag_retrieval_chunks.observe(len(chunks))

    # Early abstain based on raw distances (unchanged semantics)
    best_d = min((float(_gx(c, "distance", 1.0)) for c in chunks), default=1.0)
    if grounded_only and (not chunks or best_d > null_threshold):
        return ChatResponse(answer="I don't know.", sources=[])

    # --- 1.a) Extractive pre-gate with alpha/alpha_hits (optional stricter check) ---
    if extractive and alpha is not None:
        hits = sum(1 for c in chunks if float(_gx(c, "distance", 1.0)) <= float(alpha))
        if hits < max(0, int(alpha_hits)):
            return ChatResponse(answer="I don't know.", sources=[])

    # --- 2) Optional A3 rerank (order only) ---
    if rerank and chunks:
        chunks = rerank_chunks(req.question, chunks, rerank_lex_w)[:k]

    # --- 3) Build context + sources and an id→distance map for gates ---
    ctx_blocks: list[str] = []
    sources: list[ChatSource] = []
    id2dist: dict[str, float] = {}
    for i, c in enumerate(chunks, start=1):
        text = _gx(c, "text", "") or ""
        src = _gx(c, "source", "unknown") or "unknown"
        cid = _gx(c, "id", f"chunk::{i}") or f"chunk::{i}"
        ctx_blocks.append(f"[{i}] {text}")
        sources.append(ChatSource(index=i, id=cid, source=src, text=text))
        try:
            id2dist[cid] = float(_gx(c, "distance", 1.0) or 1.0)
        except Exception:
            id2dist[cid] = 1.0
    context = "\n\n".join(ctx_blocks)

    # --- 4) Two answer paths ---
    if extractive:
        # ===== Strict EXTRACTIVE =====
        messages = [
            {"role": "system", "content": CHAT_SYS_PROMPT},
            {"role": "user", "content": f"Question:\n{req.question}\n\nContext:\n{context}"},
        ]
        primary = _CLIENT.chat(
            model=_MODEL,
            messages=messages,
            options={"num_ctx": _NUM_CTX, "temperature": 0.0, "top_p": 0},
        )
        final_answer = (primary.get("message", {}) or {}).get("content", "").strip()

        # Deterministic A2 evidence span
        evidence = locate_best_span(final_answer, sources) if final_answer else None

        # 1) NOT_IN_CONTEXT / empty guard
        if not final_answer or final_answer.lower() == "i don't know." or evidence is None:
            return ChatResponse(answer="I don't know.", sources=sources, evidence_span=None)

        # 2) Distance gate for the evidence chunk (if requested)
        if span_max_distance is not None:
            ev_dist = id2dist.get(evidence.doc_id, 1.0)
            if ev_dist > span_max_distance:
                return ChatResponse(answer="I don't know.", sources=sources, evidence_span=None)

        # 3) Sentence-first + window support (soften for WHERE/WHEN)
        src_text = next((s.text for s in sources if s.id == evidence.doc_id), "")
        if src_text:
            sL, sR = _sentence_bounds(src_text, evidence.start, evidence.end)
            sent_support = _overlap_score(req.question, src_text[sL:sR])

            L = max(0, evidence.start - support_window)
            R = min(len(src_text), evidence.end + support_window)
            win_support = _overlap_score(req.question, src_text[L:R])

            ql = (req.question or "").strip().lower()
            soften = ql.startswith("where") or ql.startswith("when")
            gate = min(support_min, 0.05) if soften else support_min

            if max(sent_support, win_support) < gate:
                return ChatResponse(answer="I don't know.", sources=sources, evidence_span=None)

        return ChatResponse(
            answer=final_answer if final_answer else "I don't know.",
            sources=sources,
            evidence_span=evidence,
        )

    else:
        # ===== GENERATIVE (but grounded) =====
        messages = [
            {"role": "system", "content": CHAT_SYS_PROMPT},
            {"role": "user", "content": f"Question:\n{req.question}\n\nContext:\n{context}"},
        ]
        primary = _CLIENT.chat(
            model=_MODEL,
            messages=messages,
            options={"num_ctx": _NUM_CTX, "temperature": temperature, "top_p": 0},
        )
        final_answer = (primary.get("message", {}) or {}).get("content", "").strip()

        # Grounding: verify answer is recoverable from sources (span-based),
        # and if model abstains, try an extractive RESCUE using top-1 chunk.
        evidence = None
        if grounded_only:
            if final_answer and final_answer.lower() != "i don't know.":
                evidence = locate_best_span(final_answer, sources)
            if (not final_answer or final_answer.lower() == "i don't know." or not evidence):
                top_text = sources[0].text if sources else ""
                if top_text:
                    rescued = _refine_text(req.question, top_text) or ""
                    if rescued and rescued.lower() != "i don't know.":
                        ev2 = locate_best_span(rescued, sources)
                        if ev2:
                            cleaned = rescued.strip()
                            return ChatResponse(
                                answer=cleaned if cleaned else "I don't know.",
                                sources=sources,
                                evidence_span=ev2,
                            )
                return ChatResponse(answer="I don't know.", sources=sources)

        return ChatResponse(
            answer=final_answer if final_answer else "I don't know.",
            sources=sources,
            evidence_span=evidence,
        )

# --- Debug routes ---
@app.get("/debug/search")
async def debug_search(q: str, k: int = 5, max_distance: float = 0.45):
    return search(q, k, max_distance)

@app.get("/debug/samples")
async def debug_samples(n: int = 4):
    return get_sample_chunks(n)
