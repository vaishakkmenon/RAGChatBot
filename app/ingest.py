import os
import re
import hashlib
import logging
from typing import List, Dict, Optional
from .retrieval import add_documents
from .settings import settings
from fastapi import HTTPException
from .metrics import rag_ingested_chunks_total, rag_ingest_skipped_files_total

ALLOWED_EXT = {".txt", ".md"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB max per file
BATCH_SIZE = 500

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
    Split text into overlapping chunks, preferring natural boundaries (paragraphs/sentences).
    
    Args:
        text (str): Raw text to chunk.
        chunk_size (int): Target maximum size of each chunk (in characters).
        overlap (int): Number of characters to overlap between consecutive chunks.
    
    Returns:
        list[str]: List of non-empty text chunks.
    """
    if not text.strip():
        return []

    # First split into paragraphs, then sentences
    paragraphs = re.split(r"\n\s*\n", text)
    units: list[str] = []
    for para in paragraphs:
        # Keep paragraph if short, otherwise split by sentences
        if len(para) <= chunk_size:
            units.append(para.strip())
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for s in sentences:
                if s.strip():
                    units.append(s.strip())

    # Now pack units into chunks ~chunk_size long
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not unit:
            continue
        if len(current) + len(unit) + 1 <= chunk_size:
            current = (current + " " + unit).strip()
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)

    # Apply overlap: re-slice chunks into overlapping windows
    overlapped: list[str] = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        overlapped.append(chunk)
        # Add overlap with next chunk if possible
        if overlap > 0 and i + 1 < len(chunks):
            actual_ov = min(overlap, len(chunk))
            overlap_text = chunk[-actual_ov:]
            next_part = chunks[i + 1][: chunk_size - actual_ov]
            combined = overlap_text + next_part
            if combined.strip():
                overlapped.append(combined.strip())

    # Deduplicate and clean
    final = []
    seen = set()
    for ch in overlapped:
        ch = ch.strip()
        if ch and ch not in seen:
            seen.add(ch)
            final.append(ch)

    return final

def ingest_paths(paths: Optional[List[str]] = None) -> int:
    """
    Ingests text files from the given paths, chunking them and adding to the retrieval database.
    Skips files that are too large, unreadable, or produce no valid chunks.
    De-duplicates chunk TEXTS within this ingestion run via SHA-256, and adds to Chroma in batches of 500.
    
    Args:
        paths (Optional[List[str]]): List of file or directory paths to ingest. Defaults to settings.docs_dir if None.
    Returns:
        int: Total number of text chunks successfully ingested.
    """
    base_paths = paths or [settings.docs_dir]
    files = find_files(base_paths)

    docs_batch: List[Dict] = []
    added_total = 0
    duplicates_total = 0
    seen_hashes: set[str] = set()

    for fp in files:
        # stat
        try:
            file_size = os.path.getsize(fp)
        except Exception as e:
            logging.warning(f"Could not stat file {fp}: {e}")
            continue

        # guards
        if file_size > MAX_FILE_SIZE:
            logging.warning(f"Skipping {fp}: file too large ({file_size} bytes)")
            rag_ingest_skipped_files_total.labels(reason="too_large").inc()
            continue

        if os.path.splitext(fp)[1].lower() not in ALLOWED_EXT:
            logging.warning(f"Skipping {fp}: invalid extension")
            rag_ingest_skipped_files_total.labels(reason="invalid_ext").inc()
            continue

        # read
        try:
            text = read_text(fp)
        except Exception as e:
            logging.warning(f"Could not read {fp}: {e}")
            continue

        # chunk
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            logging.info(f"{fp}: produced no valid chunks (empty or whitespace).")
            continue

        # per-file counters
        file_added = 0
        file_dupes = 0

        # enqueue chunks with de-duplication
        for idx, ch in enumerate(chunks):
            # hash on normalized text to catch trivial whitespace dupes
            norm = ch.strip()
            if not norm:
                continue
            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                file_dupes += 1
                duplicates_total += 1
                logging.info(f"Skipped duplicate chunk {fp}:{idx} (sha256={h[:8]})")
                continue

            seen_hashes.add(h)
            docs_batch.append({"id": f"{fp}:{idx}", "text": norm, "source": fp})
            file_added += 1

            # flush batch
            if len(docs_batch) >= BATCH_SIZE:
                add_documents(docs_batch)
                rag_ingested_chunks_total.inc(len(docs_batch))
                added_total += len(docs_batch)
                logging.info(f"Added batch of {len(docs_batch)} chunks to Chroma.")
                docs_batch.clear()

        logging.info(f"{fp}: added {file_added} chunks, skipped {file_dupes} duplicates")

    # flush any remaining
    if docs_batch:
        add_documents(docs_batch)
        rag_ingested_chunks_total.inc(len(docs_batch))
        added_total += len(docs_batch)
        logging.info(f"Added final batch of {len(docs_batch)} chunks to Chroma.")

    logging.info(
        f"Ingest complete: added {added_total} chunks from {len(files)} files; "
        f"skipped {duplicates_total} duplicate chunks."
    )
    return added_total