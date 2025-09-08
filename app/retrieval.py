from __future__ import annotations

import os
import random
import logging
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

try:
    # We still import settings for projects that provide it,
    # but every attribute is safely defaulted via _get().
    from .settings import settings  # type: ignore
except Exception:  # pragma: no cover - extremely defensive
    settings = object()  # sentinel with no attributes

logger = logging.getLogger(__name__)

# ------------------------------
# Config helpers with safe fallbacks
# ------------------------------

def _get(name: str, default: str) -> str:
    """Get from settings.<name>, else ENV[name.upper()], else default."""
    env_key = name.upper()
    # 1) settings.<name>
    try:
        val = getattr(settings, name)  # type: ignore[attr-defined]
        if val is not None and str(val).strip() != "":
            return str(val)
    except Exception:
        pass
    # 2) ENV
    env_val = os.getenv(env_key)
    if env_val is not None and env_val.strip() != "":
        return env_val
    # 3) default
    return default

EMBED_MODEL: str = _get("embed_model", "BAAI/bge-small-en-v1.5")
CHROMA_PATH: str = _get("chroma_dir", "./data/chroma")
COLLECTION_NAME: str = _get("collection_name", "rag_docs")
TOP_K_DEFAULT: int = int(_get("top_k", "4"))
MAX_DISTANCE_DEFAULT: float = float(_get("max_distance", "0.45"))

# ------------------------------
# Embeddings & Chroma collection
# ------------------------------

_embed = SentenceTransformerEmbeddingFunction(EMBED_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=ChromaSettings(allow_reset=False))
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    # use cosine for SBERT/BGE models
    metadata={"hnsw:space": "cosine"},
    embedding_function=_embed, # type: ignore
)

# ------------------------------
# Public API
# ------------------------------

def add_documents(docs: List[dict]) -> None:
    """Add chunks to the vector store.
    Each doc must have keys: id, text, source.
    """
    if not docs:
        return
    _collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )

# BGE v1.5 query instruction improves retrieval if you're using those models
BGE_V15_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def _with_query_instruction(q: str) -> str:
    name = (EMBED_MODEL or "").lower()
    if "bge" in name and "v1.5" in name and "m3" not in name:
        return BGE_V15_QUERY_PREFIX + q
    return q

def search(
    query: str,
    k: Optional[int] = None,
    max_distance: Optional[float] = None,
) -> List[dict]:
    """Retrieve top-k chunks within a distance threshold.

    Args:
        query: user text
        k: number of results (defaults to TOP_K_DEFAULT)
        max_distance: filter threshold (defaults to MAX_DISTANCE_DEFAULT)

    Returns: list of {id, text, source, distance}
    """
    q = (query or "").strip()
    if not q:
        logger.info("Search skipped: empty query")
        return []

    k = k or TOP_K_DEFAULT
    md = MAX_DISTANCE_DEFAULT if max_distance is None else max_distance

    qtext = _with_query_instruction(q)

    res = _collection.query(
        query_texts=[qtext],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    # Defensive: if docs/metas missing but ids present, fetch via get()
    if (not docs or not metas) and ids:
        fetched = _collection.get(ids=ids, include=["documents", "metadatas"])
        docs = fetched.get("documents") or docs
        metas = fetched.get("metadatas") or metas

    out: List[dict] = []
    for i, t, m, d in zip(ids, docs, metas, dists):
        try:
            if d is not None and float(d) <= float(md):
                src = m.get("source", "unknown") if isinstance(m, dict) else "unknown"
                out.append({"id": i, "text": t, "source": src, "distance": d})
        except Exception:
            continue

    logger.info("Search '%s': %d/%d results within max_distance %.3f", query, len(out), k, md)
    return out

def get_sample_chunks(n: int = 10) -> List[dict]:
    """Return up to n example chunks for UI demos/tests."""
    n = max(1, n)
    count = _collection.count()
    if count == 0:
        return []

    start = random.randint(0, max(0, count - n))
    res = _collection.get(include=["documents", "metadatas"], limit=n, offset=start)

    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    out: List[dict] = []
    for idx, (t, m) in enumerate(zip(docs, metas), start=start):
        src = m.get("source", "unknown") if isinstance(m, dict) else "unknown"
        out.append({"id": f"sample::{idx}", "text": t, "source": src, "distance": None})
    return out