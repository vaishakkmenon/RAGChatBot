# app/metrics.py
from prometheus_client import Counter, Histogram

# Ingestion metrics
rag_ingested_chunks_total = Counter(
    "rag_ingested_chunks_total",
    "Total number of document chunks successfully ingested"
)

rag_ingest_skipped_files_total = Counter(
    "rag_ingest_skipped_files_total",
    "Number of files skipped during ingestion",
    ["reason"]  # label: e.g. "too_large", "invalid_ext", "outside_docs_dir"
)

# Retrieval metrics
rag_retrieval_chunks = Histogram(
    "rag_retrieval_chunks",
    "Number of chunks retrieved per query",
    buckets=[0, 1, 2, 4, 8, 16]
)

# LLM metrics
rag_llm_request_total = Counter(
    "rag_llm_request_total", "Total /chat LLM requests", ["status", "model"]
)
rag_llm_latency_seconds = Histogram(
    "rag_llm_latency_seconds", "LLM latency (s)", ["status", "model"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60)
)