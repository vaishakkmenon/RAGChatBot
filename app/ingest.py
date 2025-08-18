import os
import logging
from typing import List, Dict, Optional
from .retrieval import add_documents
from .settings import settings
from fastapi import HTTPException

ALLOWED_EXT = {".txt", ".md"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB max per file

def read_text(path: str) -> str:
    """
    Read and return the full text content of a file using UTF-8 encoding (ignores errors).
    
    Args:
        path (str): File path to read.
    Returns:
        str: File contents as a single string.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def find_files(base_paths: List[str]) -> List[str]:
    """
    Recursively find all allowed text files (.md, .txt) in the given paths.
    Accepts both files and directories.
    
    Args:
        base_paths (List[str]): List of file or directory paths to search.
    Returns:
        List[str]: List of discovered file paths.
    Raises:
        HTTPException: If no files are found to ingest.
    """
    files = []
    base_docs_dir = os.path.abspath(settings.docs_dir)

    for base in base_paths:
        abs_base = os.path.abspath(base)

        if not abs_base.startswith(base_docs_dir):
            logging.warning(f"Skipping {base}: outside docs_dir {base_docs_dir}")
            continue

        if os.path.isfile(abs_base):
            ext = os.path.splitext(abs_base)[1].lower()
            if ext in ALLOWED_EXT:
                files.append(abs_base)
            else:
                logging.warning(f"Skipping {abs_base}: invalid extension")
        else:
            for root, _, fs in os.walk(abs_base):
                for name in fs:
                    fp = os.path.join(root, name)
                    ext = os.path.splitext(name)[1].lower()
                    abs_fp = os.path.abspath(fp)
                    if abs_fp.startswith(base_docs_dir) and ext in ALLOWED_EXT:
                        files.append(abs_fp)
                    else:
                        logging.warning(f"Skipping {fp}: invalid or outside docs_dir")

    if not files:
        raise HTTPException(status_code=400, detail="No valid .txt or .md files found to ingest inside docs_dir.")
    return files

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks for ingestion.
    
    Args:
        text (str): The raw text to chunk.
        chunk_size (int): Maximum size (in characters) of each chunk.
        overlap (int): Number of characters to overlap between consecutive chunks.
    Returns:
        list[str]: List of non-empty text chunks.
    """
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
    """
    Ingests text files from the given paths, chunking them and adding to the retrieval database.
    Skips files that are too large, unreadable, or produce no valid chunks.
    
    Args:
        paths (Optional[List[str]]): List of file or directory paths to ingest. Defaults to settings.docs_dir if None.
    Returns:
        int: Total number of text chunks successfully ingested.
    """
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