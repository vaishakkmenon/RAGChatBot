import random
import logging
import chromadb
from .settings import settings
from typing import List, Optional
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

# Instantiate the embedding function singleton
_embed = SentenceTransformerEmbeddingFunction(settings.embed_model)

# Create persistent Chroma client
_client = chromadb.PersistentClient(
    path=settings.chroma_dir,
    settings=ChromaSettings(anonymized_telemetry=False)
)

# Create or get collection (name="docs") with custom embedding
_collection = _client.get_or_create_collection(
    name="docs",
    embedding_function=_embed # type: ignore
)

def add_documents(docs: List[dict]) -> None:
    """
    Add a list of document dicts to the persistent Chroma collection.
    
    Each document should contain:
        - id (str): unique chunk identifier
        - text (str): content for embedding/search
        - source (str): original file path
    """
    if not docs:
        return
    _collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )

# --- BGE v1.5 query instruction (applied to queries only) ---
BGE_V15_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def _with_query_instruction(q: str) -> str:
    """
    Apply the official BGE v1.5 query instruction prefix when appropriate.

    This affects the **query embedding only** and leaves previously-ingested
    document embeddings unchanged (no re-ingest required).

    Applied when:
        - settings.embed_model contains "bge" and "v1.5"
        - and does NOT contain "m3" (BGE-M3 uses different conventions)

    Args:
        q (str): Raw user query.

    Returns:
        str: Possibly-prefixed query string suitable for BGE v1.5 retrieval.
    """
    name = (settings.embed_model or "").lower()
    if "bge" in name and "v1.5" in name and "m3" not in name:
        return BGE_V15_QUERY_PREFIX + q
    return q


def search(
    query: str,
    k: Optional[int] = None,
    max_distance: Optional[float] = None
) -> List[dict]:
    """
    Retrieve top-k most similar document chunks to the query (BGE-aware).

    Args:
        query (str): Search string.
        k (int, optional): Number of results to return. Defaults to settings.top_k.
        max_distance (float | None): Maximum vector distance allowed (lower is stricter, e.g. 0.25–0.4).
            If None, uses settings.max_distance (or 0.7 if not present).

    Returns:
        List[dict]: Matching chunks with id, text, source, and distance.

    Notes:
        - If settings.embed_model is a BGE v1.5 model (e.g., BAAI/bge-*-en-v1.5),
          the query is prefixed with the recommended instruction to improve retrieval:
          "Represent this sentence for searching relevant passages: "
          Document embeddings remain unchanged.
        - Chroma returns distances where smaller is more similar. Tune `max_distance`
          for your model/data; 0.6–0.8 is a practical starting range with BGE.
        - If some backends omit documents/metadatas in `query(...)`, a defensive
          fallback fetch via `get(...)` is performed using the returned IDs.
    """
    # Early exit on empty/whitespace-only queries
    q = (query or "").strip()
    if not q:
        logger.info("Search skipped: empty query")
        return []

    k = k or settings.top_k
    md = max_distance if max_distance is not None else getattr(settings, "max_distance", 0.7)

    qtext = _with_query_instruction((query or "").strip())
    if not qtext:
        logger.info("Search skipped: empty query")
        return []

    res = _collection.query(
        query_texts=[qtext],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    ) or {}

    ids   = (res.get("ids") or [[]])[0]
    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    if (not docs or not metas) and ids:
        fetched = _collection.get(ids=ids, include=["documents", "metadatas"]) or {}
        docs  = fetched.get("documents") or docs
        metas = fetched.get("metadatas") or metas

    out = []
    for i, t, m, d in zip(ids, docs, metas, dists):
        if d is not None and d <= md:  # <- just guard None and compare
            src = m.get("source", "unknown") if isinstance(m, dict) else "unknown"
            out.append({"id": i, "text": t, "source": src, "distance": d})

    logger.info(f"Search '{query}': {len(out)}/{k} results within max_distance {md}")
    return out

def get_sample_chunks(n: int = 10) -> List[dict]:
    """
    Get a sample of document chunks from the collection for debugging or inspection.
    
    Args:
        n (int): Number of random sample chunks to return (default 10).
    
    Returns:
        List[dict]: Each dict includes 'id', 'source', and a preview of 'text' (max 120 chars).
    """
    res = _collection.get()
    ids = res.get("ids") or []
    metadatas = res.get("metadatas") or []
    docs = res.get("documents") or []

    total = len(ids)
    if total == 0:
        return []

    sample_indices = random.sample(range(total), k=min(n, total))

    out = []
    for idx in sample_indices:
        i = ids[idx]
        meta = metadatas[idx] if idx < len(metadatas) else {}
        text = docs[idx] if idx < len(docs) else ""
        out.append({
            "id": i,
            "source": (meta.get("source") if meta else "unknown"),
            "text": (text[:120] + ("..." if text and len(text) > 120 else "")),
        })
    return out
