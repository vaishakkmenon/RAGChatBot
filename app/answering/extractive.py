from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, textwrap
import httpx

def _format_passages(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ch in enumerate(chunks, start=1):
        txt = ch.get("text", "")
        lines.append(f"[{i}] {txt}")
    return "\n\n".join(lines)

def build_extractive_messages(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = textwrap.dedent("""\
        You answer by COPYING the exact words from the passages.
        Rules:
        - Answer with the shortest contiguous exact span from the passages that answers the question.
        - Return only the span. No quotes, no punctuation added.
        - If the exact answer is absent, reply exactly: NOT_IN_CONTEXT.
        - Prefer noun phrases; avoid trailing clauses (e.g., after 'who/which/that').
    """).strip()
    user = f"""Passages:\n{_format_passages(chunks)}\n\nQuestion: {question}\nExact answer:"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

async def generate_extractive(
    question: str,
    chunks: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (text, meta). Text should be a verbatim span or NOT_IN_CONTEXT.
    """
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
    base = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    url = f"{base}/api/chat"
    messages = build_extractive_messages(question, chunks)
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "num_ctx": int(os.getenv("NUM_CTX", "2048")),
        },
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    text = (data.get("message") or {}).get("content", "") or ""
    meta = {
        "model": model,
        "prompt_tokens": data.get("prompt_eval_count"),
        "eval_tokens": data.get("eval_count"),
    }
    return text.strip(), meta
