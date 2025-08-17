import os
from typing import List, Dict, Optional
from .retrieval import add_documents
from .settings import settings

ALLOWED_EXT = {".txt", ".md"}

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def find_files(base_paths: List[str]) -> List[str]:
    files = []
    for base in base_paths:
        if os.path.isfile(base):
            files.append(base)
        else:
            for root, _, fs in os.walk(base):
                for name in fs:
                    if os.path.splitext(name)[1].lower() in ALLOWED_EXT:
                        files.append(os.path.join(root, name))
    return files

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j]
        if chunk.strip():
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def ingest_paths(paths: Optional[List[str]] = None) -> int:
    base_paths = paths or [settings.docs_dir]
    files = find_files(base_paths)
    docs: List[Dict] = [] 
    for fp in files:
        text = read_text(fp)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        for idx, ch in enumerate(chunks):
            docs.append({
                "id": f"{fp}:{idx}",
                "text": ch,
                "source": fp,
            })
    add_documents(docs)
    return len(docs)