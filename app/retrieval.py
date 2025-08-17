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
    if not docs:
        return
    _collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )

def search(query: str, k: Optional[int] = None, min_score: float = 0.35) -> List[dict]:
    k = k or settings.top_k
    
    res = _collection.query(query_texts=[query], n_results=k, include=["distances"]) or {}
    docs = (res.get("documents") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    
    out = []
    for i, t, m, d in zip(ids, docs, metas, dists):
        if d <= min_score:
            out.append({"id": i, "text": t, "source": m.get("source", "unknown"), "distance": d})
    logger.info(f"Search '{query}': {len(out)}/{k} results above min_score {min_score}")
    return out

def get_sample_chunks(n=10):
    res = _collection.get(limit=n)
    ids = res.get("ids") or []
    metadatas = res.get("metadatas") or []
    docs = res.get("documents") or []
    return [
        {
            "id": i,
            "source": (meta.get("source") if meta else "unknown"),
            "text": (text[:120] + ("..." if text and len(text) > 120 else ""))
        }
        for i, meta, text in zip(ids, metadatas, docs)
    ]