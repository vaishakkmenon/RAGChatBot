import os
import logging
from typing import List, Dict, Optional
from .retrieval import add_documents
from .settings import settings
from fastapi import HTTPException

ALLOWED_EXT = {".txt", ".md"}
MAX_FILE_SIZE = 5 * 1024 * 1024

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
        try:
            file_size = os.path.getsize(fp)
        except Exception as e:
            logging.warning(f"Could not stat file {fp}: {e}")
            continue
        if file_size > MAX_FILE_SIZE:
            logging.warning(f"Skipping {fp}: file too large ({file_size} bytes)")
            continue

        try:
            text = read_text(fp)
        except Exception as e:
            logging.warning(f"Could not read {fp}: {e}")
            continue
        
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            logging.warning(f"File {fp} produced no valid chunks (empty or whitespace).")
            continue

        for idx, ch in enumerate(chunks):
            docs.append({
                "id": f"{fp}:{idx}",
                "text": ch,
                "source": fp,
            })
        logging.info(f"{fp}: {len(chunks)} chunks")

    add_documents(docs)
    logging.info(f"Ingested {len(docs)} chunks from {len(files)} files (skipped files may reduce this count).")
    return len(docs)