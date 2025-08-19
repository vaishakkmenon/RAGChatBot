# RAGChatBot (Local $0)

> A **self-hosted, production-style Retrieval-Augmented Generation (RAG) chatbot**.  
> Ingest local Markdown / text files, index them in **ChromaDB**, and query them using **Ollama** LLM — with citations, streaming, metrics, and hardened APIs.

- **Local-first**: No cloud keys required.
- **Production-minded**: OpenAPI docs, request size limits, JSON logging, API key auth, Prometheus metrics.
- **Observable**: `/metrics` for latency and request counts.
- **Developer-friendly**: Docker-native CI, tests, and a minimal **Streamlit** frontend.

---

## ✨ Features

- Document ingestion (`.md`, `.txt`) with chunking + overlap
- Similarity search (cosine distance) with **max distance** cutoff
- Chat with **numbered citations** to sources
- **Streaming** token responses via SSE
- Prometheus metrics: retrieval counts, LLM latency, request totals
- Minimal **Streamlit** UI for quick demos

---

## 🧱 Stack

**FastAPI** · **Uvicorn** · **Ollama** · **SentenceTransformers** · **ChromaDB** · **Prometheus** · **Streamlit**  
Container base: **Chainguard Python** (non-root, slim)

---

## 🚀 Quick Start (Docker)

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- (Optional) [Docker Compose](https://docs.docker.com/compose/)

### 1) Clone & configure
```bash
git clone https://github.com/your-org-or-user/ragchatbot.git
cd ragchatbot
cp .env.example .env
```

Ensure your `.env` has an API key:
```env
API_KEY=dev
```

### 2) Start the API (compose)
```bash
docker compose up -d --build
# subsequent runs:
# docker compose up -d
```

Or build the runtime image directly:
```bash
docker build -t ragchatbot:app .
docker run --rm -p 8000:8000 -e API_KEY=dev ragchatbot:app
```

### 3) Verify
```bash
curl http://127.0.0.1:8000/
# => {"ok": true, "message": "Hello from RAGChatBot"}
```

OpenAPI docs: **http://localhost:8000/docs**

---

## 📥 Ingest Documents

Place files under `./data/docs` or pass explicit paths.

```bash
curl -X POST "http://localhost:8000/ingest"   -H "X-API-Key: dev"   -H "Content-Type: application/json"   -d '{"paths": ["./data/docs"]}'
```

- Accepts `.md` and `.txt`
- Skips oversized files (configurable)
- Chunks stored in **ChromaDB** for retrieval

---

## 💬 Ask a Question

Non-streaming (returns JSON):
```bash
curl -X POST "http://localhost:8000/chat?max_distance=0.65"   -H "X-API-Key: dev"   -H "Content-Type: application/json"   -d '{"question": "What is RAG?", "top_k": 4}'
```

Streaming (SSE):
```bash
curl -N -X POST "http://localhost:8000/chat?stream=true&max_distance=0.65"   -H "X-API-Key: dev"   -H "Content-Type: application/json"   -d '{"question": "Summarize the docs", "top_k": 4}'
```

Health:
```bash
curl "http://localhost:8000/health/ollama"
```

Debug (no LLM, preview chunks):
```bash
curl "http://localhost:8000/debug-search?q=embedding&k=4&max_distance=0.65"
curl "http://localhost:8000/debug-ingest?n=10"
```

---

## 🖥️ Minimal Frontend (Streamlit)

A simple demo UI is included.

```bash
# from repo root
pip install streamlit requests   # or use a separate venv
API_KEY=dev RAG_API_BASE=http://localhost:8000 streamlit run frontend/app.py
```

Open **http://localhost:8501**:
- Enter your API key (`dev` by default)
- Ask a question
- See answer + sources (filenames and snippets)

---

## 📊 Metrics

Prometheus-compatible endpoint:
```
GET /metrics
```

Exposes:
- `rag_retrieval_chunks` — number of chunks used per query
- `rag_llm_request_total{status=...}` — total LLM requests by outcome
- `rag_llm_latency_seconds` — LLM latency histogram

Pair with **Grafana** for dashboards (export the `/metrics` target).

---

## ⚙️ Configuration

All via `.env` (with safe defaults):

| Key               | Description                                   | Example                                 |
|-------------------|-----------------------------------------------|-----------------------------------------|
| `API_KEY`         | Required API key for POST endpoints           | `dev`                                   |
| `OLLAMA_HOST`     | Ollama server URL                              | `http://127.0.0.1:11434`                |
| `OLLAMA_MODEL`    | Model name                                     | `llama3.1:8b-instruct-q4_K_M`           |
| `NUM_CTX`         | Context tokens for LLM                         | `2048`                                  |
| `OLLAMA_TIMEOUT`  | Upstream timeout (seconds)                     | `60`                                    |
| `MAX_BYTES`       | Max request size (bytes)                       | `32768`                                 |
| `CHROMA_DIR`      | ChromaDB persistence dir                       | `./data/chroma`                         |
| `DOCS_DIR`        | Default docs dir to ingest                     | `./data/docs`                           |
| `TOP_K`           | Default retrieval top-k                        | `4`                                     |
| `CHUNK_SIZE`      | Chunk size (characters)                        | `600`                                   |
| `CHUNK_OVERLAP`   | Overlap between chunks                         | `120`                                   |
| `EMBED_MODEL`     | Embedding model name                           | `BAAI/bge-small-en-v1.5`                |
| `CORS_ORIGIN`     | Allowed origin for browser apps                | `http://localhost:3000`                 |

---

## 🔐 Security

- API key required for **POST** endpoints via `X-API-Key`
- Body size limit (413) via middleware
- CORS restricted to your frontend origin
- Non-root containers; minimal base images

> For public deployments, also consider rate limiting, auth providers, and TLS termination.

---

## 🧪 Testing & CI (Docker-native)

Unit tests (mocked Ollama) run inside a dedicated **test stage** image.

```bash
# build test stage and run tests
docker build --target test -t ragchatbot:test .
docker run --rm -e API_KEY=dev ragchatbot:test

# run Ruff lint
docker run --rm ragchatbot:test /opt/venv/bin/ruff check .
```

GitHub Actions:
- Builds the **test** stage (no Ollama needed)
- Passes `API_KEY` as a secret
- Runs unit tests + Ruff

---

## 🖼️ Screenshots

> _Add your screenshots to `docs/screenshots/` and update these paths:_

- **Frontend UI**  
  ![Frontend](docs/screenshots/frontend.png)

- **OpenAPI Docs**  
  ![OpenAPI](docs/screenshots/openapi.png)

---

## 🧭 Why this matters

This project shows end-to-end **engineering maturity**:

- **Systems design**: local, reproducible, secure service
- **ML ops**: embeddings + retrieval + LLM with citations
- **Platform**: observability, CI, and container best practices
- **UX**: minimal frontend to demonstrate value quickly

Use it as:
- A personal knowledge assistant over your notes/docs
- A portfolio piece demonstrating production-readiness
- A foundation for more advanced RAG (re-ranking, evaluators, structured outputs)

---

## 📄 License

MIT (or your choice). See `LICENSE`.