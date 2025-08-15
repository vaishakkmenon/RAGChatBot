import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
    num_ctx: int = int(os.getenv("NUM_CTX", 2048))
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", 60))
    max_bytes: int = int(os.getenv("MAX_BYTES", 32768))

    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    docs_dir: str = os.getenv("DOCS_DIR", "./data/docs")

    top_k: int = int(os.getenv("TOP_K", 4))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 600))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

settings = Settings()