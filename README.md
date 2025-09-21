# FastAPI RAG Chatbot (SQuADv2 Demo)

A compact Retrieval-Augmented Generation (RAG) chatbot built with FastAPI. It includes health/metrics, ingestion, retrieval + reranker (A3), extractive/generative answerers, evidence span snapping (A2), and abstention gates — with evaluation scripts and tuning grids for SQuADv2.

## Highlights

- **Endpoints**: `/health`, `/metrics`, `/ingest`, `/debug/samples`, `/debug/search`, `/rc` (reading-comprehension over given text), `/chat` (full RAG; extractive or generative)
- **Safety/Calibration**: Early retrieval null-gate, extractive pre-gate (alpha, alpha_hits), span distance/support gates (span_max_distance, support_min/support_window)
- **Reranker (A3)**: Lexical/semantic blend by default; can switch to hosted or local cross-encoder later
- **Reproducible evals**: Windows .bat to run sweeps, write JSON summaries, build leaderboards and charts

## Repository Structure

```
.
├─ main.py                   # FastAPI app: routes, middleware, CORS, metrics
├─ settings.py               # knobs via environment variables
├─ retrieval.py              # retrieve + optional rerank
├─ extractive.py             # span finding, support windows, gates
├─ ingest.py, models.py, metrics.py, api_key.py, logging.py, max_size.py
├─ scripts/
│  ├─ squad_eval.py          # SQuADv2 evaluation with progress + JSON report
│  ├─ retrieval_eval.py      # retrieval-only metrics
│  ├─ tune_eval.bat          # sweep + leaderboard builder
│  └─ build_leaderboards.py  # combine JSONs -> CSV + scatter plot
├─ data/
│  └─ squad/dev-v2.0.json    # place SQuADv2 dev set here
├─ results/
│  └─ YYYYMMDD_HHMMSS/       # per-run JSONs created by scripts
└─ docs/
   ├─ pareto_scatter.png
   └─ combined_eval_table.csv
```

## Quick Start (Windows CMD)

### 1. One-time Environment Setup

```cmd
set URL=http://127.0.0.1:8000
set KEY=my-dev-key-1
set DATASET=%CD%\data\squad\dev-v2.0.json
```

### 2. Sanity Checks

```cmd
curl -s "%URL%/health"
curl -s "%URL%/metrics" | more
```

### 3. Optional Ingestion

If your app supports loading documents:

```cmd
curl -s -X POST "%URL%/ingest" ^
  -H "X-API-Key: %KEY%" -H "Content-Type: application/json" ^
  --data-binary "{\"paths\":[]}"
```

## Debug Helpers

### Random Sample Chunks

```cmd
curl -s -H "X-API-Key: %KEY%" "%URL%/debug/samples?n=3"
```

### Search Smoke Test

```cmd
curl -s -H "X-API-Key: %KEY%" "%URL%/debug/search?q=Normans&k=5&max_distance=0.60"
```

## API Usage

### RC Endpoint (Pure Extractive Over Provided Context)

```cmd
curl -s -X POST "%URL%/rc" ^
  -H "X-API-Key: %KEY%" -H "Content-Type: application/json" ^
  --data-binary "{\"question\":\"Who were the Normans descended from?\", \"context\":\"The Normans ... were descended from Norse raiders and pirates from Denmark, Iceland and Norway.\"}"
```

### Chat Endpoint (Two Recommended Presets)

#### A) Generative Grounded (Fast, Well-Calibrated)

```
%URL%/chat?extractive=0&grounded_only=1&top_k=5&max_distance=0.60&null_threshold=0.25&rerank=1&rerank_lex_w=0.5&temperature=0
```

#### B) Extractive (Span-First)

```
%URL%/chat?extractive=1&grounded_only=1&top_k=3&max_distance=0.60&null_threshold=0.60&rerank=1&rerank_lex_w=0.5&alpha=0.50&alpha_hits=2&support_min=0.30&support_window=96&span_max_distance=0.60&temperature=0
```

**Notes**: These reflect your best runs - generative is faster and better-calibrated; extractive slightly higher overall F1.

## Evaluation

Run a 500-example SQuADv2 sweep and generate leaderboards. If you already have results JSONs, you can skip to the "Build tables and graph" section.

### Example: Generative Grounded Highlight Run

```cmd
python scripts\squad_eval.py ^
  --dataset "%DATASET%" --host "%URL%" --api-key "%KEY%" ^
  --grounded-only ^
  --top-k 5 --max-distance 0.60 --null-threshold 0.25 ^
  --temperature 0 --workers 4 --timeout 180 --limit 500 ^
  --progress --progress-interval 50 ^
  --out results\gen_k5_md060_nt0p25_rerank1.json ^
  --rerank --rerank-lex-w 0.5
```

### Example: Extractive Highlight Run

```cmd
python scripts\squad_eval.py ^
  --dataset "%DATASET%" --host "%URL%" --api-key "%KEY%" ^
  --extractive --grounded-only ^
  --rerank --rerank-lex-w 0.5 ^
  --top-k 3 --max-distance 0.60 --null-threshold 0.60 ^
  --alpha 0.50 --alpha-hits 2 ^
  --support-min 0.30 --support-window 96 ^
  --span-max-distance 0.60 ^
  --temperature 0 --workers 4 --timeout 180 --limit 500 ^
  --progress --progress-interval 50 ^
  --out results\ex_sd0p60.json
```

## Build Tables and Graph (Leaderboards + Pareto Scatter)

**Prerequisites**: Python, pandas, matplotlib installed. Place `scripts\build_leaderboards.py` in your repo.

### Auto-detect Newest Results Subfolder

```cmd
python scripts\build_leaderboards.py
```

### Or Specify a Folder Explicitly

```cmd
python scripts\build_leaderboards.py --folder results\YYYYMMDD_HHMMSS
```

### Outputs

- `docs\combined_eval_table.csv` - One row per JSON; F1, EM, NoAns, p50_ms, and all key knobs
- `docs\pareto_scatter.png` - Overall F1 vs NoAns Accuracy; point size ~ 1/sqrt(p50_ms)

The script also prints three leaderboards to the console:
- Top 10 by Overall F1
- Top 10 by NoAns Accuracy  
- Top 10 by Answerable F1

## Results Snapshot (SQuADv2 Dev Slice, k ≤ 5)

### Best Generative (Grounded)
- **Overall F1**: ~0.52
- **EM**: ~0.49
- **NoAns**: ~0.74
- **p50**: ~614 ms
- **Params**: `top_k=5`, `max_distance=0.60`, `null_threshold=0.25`, `rerank=1`, `rerank_lex_w=0.5`, `grounded_only=1`, `temperature=0`

### Best Extractive (Span-First)
- **Overall F1**: ~0.53
- **EM**: ~0.50
- **NoAns**: ~0.67
- **p50**: ~1250 ms
- **Params**: `top_k=3`, `max_distance=0.60`, `null_threshold=0.60`, `rerank=1`, `rerank_lex_w=0.5`, `alpha=0.50`, `alpha_hits=2`, `support_min=0.30`, `support_window=96`, `span_max_distance=0.60`, `temperature=0`

### Notes
- These are untrained RAG numbers (no weight fine-tuning)
- Generative wins on speed and abstention; extractive edges overall F1 slightly

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service status |
| GET | `/metrics` | Prometheus metrics |
| POST | `/ingest` | (Re)load data; body: `{"paths":[]}` |
| GET | `/debug/samples?n=3` | Random chunk texts |
| GET | `/debug/search?q=...` | Retrieval only; supports k and max_distance |
| POST | `/rc` | Extractive RC over provided context |
| POST | `/chat` | Full RAG |

### Key `/chat` Query Parameters

- `extractive=0|1`, `grounded_only=1`, `top_k`, `max_distance`
- `null_threshold`, `temperature`, `rerank=true|false`, `rerank_lex_w`
- **Extractive-only gates**: `alpha`, `alpha_hits`, `support_min`, `support_window`, `span_max_distance`

### Authentication

Use the header: `X-API-Key: your-key-here`

## Roadmap (Next Steps)

- **Plug-in rerankers**: Cohere/Voyage (hosted) or local cross-encoder (MiniLM/BGE); keep `?rerank=true` flag
- **Train a small cross-encoder** reranker on pairs mined from your service (scripts provided)
- **Retriever tuning** with hard negatives to lift Recall@k and answerable F1
- **Learned abstention head** for tighter calibration (optional)

## Are These Results "Good"?

For an untrained RAG over SQuADv2 with small k and a light reranker, yes:
- Overall F1 about 0.52—0.53
- NoAns about 0.67—0.74
- p50 about 0.6—1.25 s

The main headroom is answerable F1; a stronger reranker is typically the next lever.

## License and Data

- **Code**: Choose a license (for example, MIT)
- **Data**: SQuAD v2.0. Respect the original license/attribution