import asyncio
import json
import logging
import os
import socket
import time
import ollama
import re
import string

from typing import Optional, Tuple, Sequence, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .settings import settings
from .ingest import ingest_paths
from .retrieval import search, get_sample_chunks
from .answering.extractive import generate_extractive
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

logger = logging.getLogger(__name__)

# --- Ollama client & core settings ---
_CLIENT = ollama.Client(host=settings.ollama_host)
_MODEL = settings.ollama_model
_NUM_CTX = settings.num_ctx
REQUEST_TIMEOUT_S = settings.ollama_timeout
MAX_BYTES = settings.max_bytes

# --- RC behavior toggles (env-driven; keep centralized semantics in settings if you add them later) ---
RC_WINDOW_PAD = int(os.getenv("RC_WINDOW_PAD", "48"))           # IMPORTANT: local refine window size
RC_ALWAYS_SHRINK = os.getenv("RC_ALWAYS_SHRINK", "true").lower() == "true"
RC_USE_NEAREST_WORD = os.getenv("RC_USE_NEAREST_WORD", "false").lower() == "true"
RC_USE_TYPE_HINTS = os.getenv("RC_USE_TYPE_HINTS", "false").lower() == "true"

# --- System prompts (stable; keep short to reduce token overhead) ---
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

# --- FastAPI app & middleware wiring ---
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

# API key, request-size limit, and structured logging
app.add_middleware(APIKeyMiddleware)
app.add_middleware(MaxSizeMiddleware, max_bytes=MAX_BYTES)
app.add_middleware(LoggingMiddleware)

# Prometheus /metrics
Instrumentator().instrument(app).expose(app)


# ========== Helpers (RC + answer verification) ==========

def _contains_word(s: str) -> bool:
    """True if the string has at least one alphanumeric character."""
    return any(ch.isalnum() for ch in (s or ""))


def _snap_to_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Clamp to [0, len], expand if mid-token, trim whitespace,
    trim simple leading quotes/brackets, and trim trailing punctuation.
    Keep a trailing '.' if it looks like part of an acronym.
    """
    n = len(text)
    start = max(0, min(start, n))
    end = max(0, min(end, n))
    if end < start:
        start, end = end, start

    # Expand if cutting words
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

    # Handle trailing '.'
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
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Text (index within THIS text):\n{text}\n\n"
                f"Return JSON ONLY."
            ),
        },
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


def _nearest_word(text: str, anchor: int) -> tuple[int | None, int | None]:
    """Return [start, end) of the closest alphanumeric word near `anchor`."""
    n = len(text)
    i = max(0, min(anchor, n))

    # Prefer left of anchor
    j = i - 1
    while j >= 0 and not text[j].isalnum():
        j -= 1
    if j >= 0:
        l = j
        while l - 1 >= 0 and text[l - 1].isalnum():
            l -= 1
        r = j + 1
        while r < n and text[r].isalnum():
            r += 1
        return l, r

    # Else right of anchor
    j = i
    while j < n and not text[j].isalnum():
        j += 1
    if j < n:
        l = j
        r = j + 1
        while r < n and text[r].isalnum():
            r += 1
        return l, r

    return None, None


# --- Optional type-aware (WHO→PERSON) heuristics, gated by RC_USE_TYPE_HINTS ---
_PERSON_RE = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}$")

def _infer_answer_type(question: str) -> str | None:
    q = (question or "").strip().lower()
    if q.startswith("who") or " who " in q:
        return "PERSON"
    return None


def _is_person_like(s: str) -> bool:
    s = (s or "").strip()
    return bool(_PERSON_RE.match(s))


def _person_from_window(text: str, left: int, right: int) -> tuple[int | None, int | None]:
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
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nText:\n{text}\n\nReturn JSON ONLY.",
        },
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

# =========================
# A2: Deterministic span snapper helpers (pure string ops)
# =========================
_WS_RE = re.compile(r"\s+")
_TOK_RE = re.compile(r"\w+", re.UNICODE)

def _normalize_ws(s: str) -> str:
    """Collapse all whitespace to single spaces and trim."""
    return _WS_RE.sub(" ", s).strip()

def _find_span_basic(answer: str, text: str) -> Optional[Tuple[int, int]]:
    """Try exact, then case-insensitive substring match. Return (start,end) or None."""
    if not answer:
        return None
    # Exact
    idx = text.find(answer)
    if idx != -1:
        return idx, idx + len(answer)
    # Case-insensitive
    ans_ci = answer.casefold()
    txt_ci = text.casefold()
    idx = txt_ci.find(ans_ci)
    if idx != -1:
        return idx, idx + len(answer)
    return None

def _find_span_subseq(answer: str, text: str) -> Optional[Tuple[int, int]]:
    """
    Ordered token subsequence fallback.
    Tokenize with '\'w+; match tokens in order (case-insensitive).
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
        tok = m.group(0).casefold()
        if tok == ans_tokens_ci[ai]:
            matches.append((m.start(), m.end()))
            ai += 1
            if ai == len(ans_tokens_ci):
                break

    if ai != len(ans_tokens_ci) or not matches:
        return None

    start = matches[0][0]
    end = matches[-1][1]
    return start, end

def locate_best_span(answer: str, sources: Sequence[ChatSource | Dict[str, Any]]) -> Optional[EvidenceSpan]:
    """
    Iterate sources in order; try basic match then subseq. On hit, return EvidenceSpan.
    Accepts either ChatSource objects or dicts with id/text.
    """
    if not answer:
        return None

    for src in sources:
        src_id = src.id if isinstance(src, ChatSource) else str(src.get("id", ""))
        src_text = src.text if isinstance(src, ChatSource) else str(src.get("text", ""))
        if not src_text:
            continue

        # 1) exact / ci substring
        hit = _find_span_basic(answer, src_text)
        if hit is None:
            norm_answer = _normalize_ws(answer)
            if norm_answer != answer:
                hit = _find_span_basic(norm_answer, src_text)

        # 2) subsequence fallback
        if hit is None:
            hit = _find_span_subseq(answer, src_text)

        if hit is not None:
            s, e = hit
            if 0 <= s <= e <= len(src_text):
                return EvidenceSpan(
                    doc_id=src_id,
                    start=s,
                    end=e,
                    text=src_text[s:e],
                )
    return None

# --- Generic safe getter for chunks that might be dicts or SimpleNamespace/Pydantic
def _gx(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    val = getattr(obj, key, None)
    return val if val is not None else default

# --- Tiny stopword set + overlap scoring for support gating
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


# ========== Routes ==========

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

    # IMPORTANT: Timeout + metrics around the LLM call
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


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans({c: " " for c in string.punctuation}))
    return re.sub(r"\s+", " ", s).strip()


def _answer_in_sources(ans: str, src_texts: list[str]) -> bool:
    a = _normalize(ans)
    if not a:
        return False
    ctx = _normalize(" ".join(t for t in src_texts if t))
    return a in ctx


def _best_sentence(context: str, question: str) -> str:
    q = set(_normalize(question).split())
    best, best_score = "", -1
    for sent in _SENT_SPLIT.split(context):
        s = set(_normalize(sent).split())
        score = len(q & s)  # simple token overlap
        if score > best_score:
            best, best_score = sent, score
    return best


def _source_text(s) -> str:
    if isinstance(s, dict):
        return s.get("text", "")
    # Pydantic v2 preferred: attribute access, then model_dump fallback
    t = getattr(s, "text", None)
    if t is not None:
        return t
    try:
        return s.model_dump().get("text", "")
    except Exception:
        return ""

def _extractive_should_abstain(chunks, null_threshold: float | None, alpha: float | None, min_hits: int) -> bool:
    dists = []
    for c in chunks:
        # c may be a dict or SimpleNamespace depending on your code path
        dist = getattr(c, "distance", None)
        if dist is None and isinstance(c, dict):
            dist = c.get("distance")
        if dist is not None:
            dists.append(float(dist))
    if not dists:
        return True  # nothing retrieved → abstain

    min_dist = min(dists)
    # Primary rule: use the null_threshold as a min-distance cutoff
    use_alpha = (alpha is not None and min_hits > 0)
    hits = sum(1 for d in dists if (alpha is not None and d <= float(alpha))) if use_alpha else None

    # If alpha is provided, require BOTH: min_dist < null_threshold AND hits ≥ min_hits
    if null_threshold is not None:
        if use_alpha:
            return not (min_dist < float(null_threshold) and hits is not None and hits >= min_hits)
        else:
            return min_dist >= float(null_threshold)
    else:
        # No null_threshold provided; if alpha is given, use only hits criterion
        if use_alpha:
            return hits is None or hits < min_hits
        return False

# --- RAG chat with distance-based abstention ---
@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    grounded_only: bool = Query(True, description="Abstain when retrieval is weak."),
    null_threshold: float = Query(
        settings.null_threshold,
        ge=0.0,
        le=1.0,
        description="Distance above which we say 'I don't know.'",
    ),
    temperature: float = Query(0.0, ge=0.0, le=1.0),
    max_distance: float = Query(settings.max_distance, ge=0, le=1),
    extractive: bool = Query(False, description="Use strict extractive head (copy verbatim)."),
    alpha: float | None = Query(None, ge=0.0, le=1.0, description="Stricter extractive gate cutoff."),
    alpha_hits: int = Query(1, ge=0, description="Min chunks under alpha."),
    support_min: float = Query(0.15, ge=0.0, le=1.0, description="Min token-overlap between question and window around evidence (extractive only)."),
    support_window: int = Query(96, ge=16, le=512, description="Half-window size (chars) around evidence used for support check (extractive only)."),
    span_max_distance: float | None = Query(None, ge=0.0, le=1.0,
        description="Max distance allowed for the chunk that contains the evidence span (extractive only).")
):
    # Retrieve
    k = max(1, min(20, req.top_k or settings.top_k))
    chunks = search(req.question, k=k, max_distance=max_distance)

    # IMPORTANT: histogram defined WITHOUT labels → call without .labels(...)
    rag_retrieval_chunks.observe(len(chunks))

    # Early abstain if retrieval is weak (robust for dict or attr objects)
    distances = []
    for c in chunks:
        d = getattr(c, "distance", None)
        if d is None and isinstance(c, dict):
            d = c.get("distance")
        if d is not None:
            distances.append(float(d))
    best_d = min(distances) if distances else None
    if grounded_only and (not chunks or (best_d is not None and best_d > null_threshold)):
        return ChatResponse(answer="I don't know.", sources=[])

    # Build context + sources
    ctx_blocks: list[str] = []
    sources: list[ChatSource] = []
    for i, c in enumerate(chunks, start=1):
        text = _gx(c, "text", "") or ""
        src = _gx(c, "source", "unknown") or "unknown"
        cid = _gx(c, "id", f"chunk::{i}") or f"chunk::{i}"
        ctx_blocks.append(f"[{i}] {text}")
        sources.append(ChatSource(index=i, id=cid, source=src, text=text))
    context_str = "\n\n".join(ctx_blocks)
    
    # Extractive-only pre-abstain gate to protect No-Answer accuracy
    is_extractive = extractive or os.getenv("EXTRACTIVE", "0") == "1"

    # Optional stricter knobs via env; you can also expose as Query params later
    alpha_env = os.getenv("EX_ALPHA")
    alpha_eff = alpha if alpha is not None else (float(alpha_env) if alpha_env else None)
    min_hits_eff = alpha_hits if alpha_hits is not None else int(os.getenv("EX_ALPHA_HITS", "1"))
    
    # === Generate ===
    answer = "I don't know."  # default
    raw_generation = None
    t0 = time.time()
    try:
        if is_extractive:
            if _extractive_should_abstain(chunks, null_threshold, alpha_eff, min_hits_eff):
                return ChatResponse(
                    answer="I don't know.",
                    sources=sources,
                    raw_generation="NOT_IN_CONTEXT",
                    evidence_span=None,
                )
            # Extractive path: copy verbatim from passages (use RAW chunk text)
            passages = [{"id": s.id, "text": s.text} for s in sources]
            raw_generation, _meta = await generate_extractive(
                req.question,
                passages,
                temperature=temperature,
            )
            ans = (raw_generation or "").strip()
            # Treat explicit model abstain marker as null
            answer = "I don't know." if ans.upper() == "NOT_IN_CONTEXT" else (ans or "I don't know.")
        else:
            # Default generative path
            messages = [
                {"role": "system", "content": CHAT_SYS_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{req.question}\n\n"
                        f"Use ONLY the following context if relevant:\n{context_str}"
                    ),
                },
            ]
            resp = await asyncio.wait_for(
                run_in_threadpool(
                    _CLIENT.chat,
                    model=_MODEL,
                    messages=messages,
                    options={
                        "num_ctx": _NUM_CTX,
                        "temperature": temperature,
                        "top_p": 0,
                        "num_predict": settings.num_predict,
                        "stop": ["\n[", "\nUse ONLY", "\nQuestion:", "\nContext:"],
                    },
                ),
                timeout=REQUEST_TIMEOUT_S,
            )
            answer = ((resp.get("message") or {}).get("content") or "").strip() or "I don't know."

        rag_llm_request_total.labels(status="ok", model=_MODEL).inc()
        rag_llm_latency_seconds.labels(status="ok", model=_MODEL).observe(time.time() - t0)

    except asyncio.TimeoutError:
        rag_llm_request_total.labels(status="timeout", model=_MODEL).inc()
        rag_llm_latency_seconds.labels(status="timeout", model=_MODEL).observe(time.time() - t0)
        raise HTTPException(status_code=504, detail="LLM timeout")

    except Exception as e:
        rag_llm_request_total.labels(status="error", model=_MODEL).inc()
        rag_llm_latency_seconds.labels(status="error", model=_MODEL).observe(time.time() - t0)
        logger.exception("LLM /chat failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Verify grounding if requested
    clean = re.sub(r"\[\d+\]", "", answer).strip()
    if grounded_only and not (extractive or os.getenv("EXTRACTIVE", "0") == "1"):
        src_texts = [s.text for s in sources]
        ok = _answer_in_sources(clean, src_texts)
        if not ok:
            clean = "I don't know."

    # Heuristic sentence check (kept from original behavior)
    if not (extractive or os.getenv("EXTRACTIVE", "0") == "1"):
        src_texts = [_source_text(s) for s in sources]
        ctx_joined = " ".join(src_texts)
        sent = _best_sentence(ctx_joined, req.question)
        if grounded_only and _normalize(clean) not in _normalize(sent):
            clean = "I don't know."

    # === A2: evidence span (only for extractive answers) ===
    evidence = None
    if (extractive or os.getenv("EXTRACTIVE", "0") == "1") and clean != "I don't know.":
        # 1) pick span
        evidence = locate_best_span(clean, sources)
        if evidence is None:
            clean = "I don't know."
        else:
            # 2) distance gate (A2.b)
            ev_chunk = next((c for c in chunks if _gx(c, "id") == evidence.doc_id), None)
            ev_dist = float(_gx(ev_chunk, "distance", 1.0)) if ev_chunk is not None else 1.0
            if span_max_distance is not None and ev_dist > span_max_distance:
                logger.info("abstain: span_dist_gate ev_dist=%.3f > gate=%.3f doc_id=%s",
                            ev_dist, span_max_distance, evidence.doc_id)
                clean = "I don't know."
                evidence = None
            else:
                # 3) support gate
                src_text = next((s.text for s in sources if s.id == (evidence.doc_id)), "")
                if src_text:
                    L = max(0, evidence.start - support_window)
                    R = min(len(src_text), evidence.end + support_window)
                    supp = _overlap_score(req.question, src_text[L:R])
                    if supp < support_min:
                        clean = "I don't know."
                        evidence = None

    return ChatResponse(
        answer=clean,
        sources=sources,
        raw_generation=raw_generation,
        evidence_span=evidence,
    )


# --- Debug routes (unchanged API shapes) ---
@app.get("/debug/search")
async def debug_search(q: str, k: int = 5, max_distance: float = 0.45):
    return search(q, k, max_distance)

@app.get("/debug/samples")
async def debug_samples(n: int = 4):
    return get_sample_chunks(n)