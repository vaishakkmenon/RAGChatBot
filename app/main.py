import os
import ollama
from fastapi import FastAPI

app = FastAPI(title="RAGChatBot")

@app.get("/")
def root():
    return {"ok": True, "message": "Hello from RAGChatBot"}

_client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
_NUM_CTX = int(os.getenv("NUM_CTX", "2048"))

@app.post("/chat-test")
def chat_test(q: dict):
    prompt = q.get("question", "Say hello in five words.")
    resp = _client.chat(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": _NUM_CTX},
    )
    return {"answer": resp.get("message", {}).get("content", "")}