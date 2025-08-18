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

def search(
    query: str, 
    k: Optional[int] = None, 
    max_distance: float = 0.35
) -> List[dict]:
    """
    Retrieve top-k most similar document chunks to the query.
    
    Args:
        query (str): Search string.
        k (int, optional): Number of results to return. Defaults to settings.top_k.
        max_distance (float): Maximum vector distance allowed (lower is stricter, e.g. 0.25-0.4).
    
    Returns:
        List[dict]: Matching chunks with id, text, source, and distance.
    """
    k = k or settings.top_k

    res = _collection.query(query_texts=[query], n_results=k, include=["distances"]) or {}
    docs = (res.get("documents") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out = []
    for i, t, m, d in zip(ids, docs, metas, dists):
        if d <= max_distance:
            out.append({"id": i, "text": t, "source": m.get("source", "unknown"), "distance": d})
    logger.info(f"Search '{query}': {len(out)}/{k} results within max_distance  {max_distance}")
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
