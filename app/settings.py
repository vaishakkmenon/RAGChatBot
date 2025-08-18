import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    """
    Global application configuration, loaded from environment variables (with sensible defaults).
    
    Attributes:
        ollama_host (str): URL for the Ollama LLM API server.
        ollama_model (str): Name of the LLM model to use.
        num_ctx (int): Context window size for Ollama model (tokens or bytes).
        ollama_timeout (int): Max seconds to wait for LLM responses.
        max_bytes (int): Max HTTP request body size (in bytes).
        chroma_dir (str): Filesystem path for ChromaDB persistent storage.
        docs_dir (str): Filesystem path for document ingestion.
        top_k (int): Default number of top results to retrieve for context.
        chunk_size (int): Chunk size (in characters) for document splitting.
        chunk_overlap (int): Overlap (in characters) between consecutive text chunks.
        embed_model (str): Embedding model name for Chroma/SentenceTransformers.
    """
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
    num_ctx: int = int(os.getenv("NUM_CTX", 2048))
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", 60))
    max_bytes: int = int(os.getenv("MAX_BYTES", 32768))

    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    docs_dir: str = os.getenv("DOCS_DIR", "./data/docs")
    api_key: str = os.getenv("API_KEY", "")

    max_distance: float = float(os.getenv("MAX_DISTANCE", "0.7"))
    top_k: int = int(os.getenv("TOP_K", 4))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 600))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

settings = Settings()