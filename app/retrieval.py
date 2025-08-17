import chromadb
from .settings import settings
from typing import List, Optional
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


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

def search(query: str, k: Optional[int] = None) -> List[dict]:
    k = k if k is not None else settings.top_k
    
    res = _collection.query(query_texts=[query], n_results=k) or {}
    docs = (res.get("documents") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    
    out = []
    for i, t, m in zip(ids, docs, metas):
        out.append({"id": i, "text": t, "source": m.get("source", "unknown")})
    
    return out
