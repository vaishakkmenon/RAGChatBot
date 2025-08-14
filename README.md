# RAGChatBot

## Environment & Tooling

This project runs in a **containerized development environment** using [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) for service orchestration. The stack consists of:

- **Base OS & Python**: [Chainguard Python](https://github.com/chainguard-images/images/tree/main/images/python) minimal container image (`cgr.dev/chainguard/python`), providing a slim, non-root Python runtime.
- **Python Version**: 3.11 (via Chainguard `latest` tag).
- **Application Framework**: [FastAPI](https://fastapi.tiangolo.com/) for building the API service.
- **Runtime Server**: [Uvicorn](https://www.uvicorn.org/) with `--reload` enabled for development.
- **Model Server**: [Ollama](https://ollama.ai/) container hosting the `llama3.1:8b-instruct-q4_K_M` model, accessible to the API over Docker’s internal network.
- **Environment Variables**: Managed via a `.env` file mounted into the API container (contains model, embedding, and storage configuration).
- **Data Storage**:
  - **Documents & ChromaDB Index** stored under `/workspace/data/docs` and `/workspace/data/chroma` (bind-mounted from the host for persistence and to work with a `read_only` root filesystem).
  - **Ollama Models** persisted in a named Docker volume (`ollama_models`).
- **Security & Hardening**:
  - Runs as a non-root user (`65532:65532`)
  - `read_only: true` root filesystem with `tmpfs` at `/tmp`
  - Dropped all Linux capabilities (`cap_drop: ["ALL"]`)
  - `no-new-privileges:true` security option
- **Development Workflow**:
  - Hot-reloading via Uvicorn
  - Source code mounted directly from the host for instant updates without rebuilding
  - Rebuild only when dependencies or Dockerfile change

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ragchatbot.git
cd ragchatbot
```

### 2. Create the `.env` file
Copy the example environment file and adjust values if needed:
```bash
cp .env.example .env
```

### 3. Start the containers
Build and start the API and Ollama services:
```bash
docker compose up -d --build
```
> On subsequent runs, you can skip --build unless dependencies or the Dockerfile change:
```bash
docker compose up -d
```

### 4. Verify the API
Once the containers are running, confirm that the API service is responding.

Run:
```bash
curl http://127.0.0.1:8000/
```

You should see a JSON response similar to:
```json 
{"ok": true, "message": "Hello from RAGChatBot"}
```
If you see this message, your FastAPI server is up and running inside the container.

### 5. Stopping and Restarting

#### Stop (keep containers/images)

Temporarily stop all running services without removing them:

```bash
docker compose stop
```

Restart them later without a rebuild:

```bash
docker compose start
```

#### Stop & remove containers (keep images/volumes)

Remove the containers but keep the images and any named volumes:

```bash
docker compose down
```

#### Full cleanup (containers, images, and volumes)

Remove everything, including images and named volumes (this will delete Ollama models and ChromaDB index):

```bash
docker compose down --rmi all --volumes
```

> **Warning:** The full cleanup command will permanently delete all Ollama models and ChromaDB index data.