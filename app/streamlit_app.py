# streamlit_app.py
import json
import requests
import textwrap
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="RAGChatBot – Streamlit UI", page_icon="💬", layout="centered")

# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
api_base = st.sidebar.text_input("API Base URL", value="http://localhost:8000", help="Your FastAPI server base URL.")
api_key = st.sidebar.text_input("X-API-Key", type="password", help="Must match the key expected by your FastAPI server.")
top_k = st.sidebar.number_input("top_k", min_value=1, max_value=50, value=4, help="How many chunks to retrieve.")
max_distance = st.sidebar.slider("max_distance", min_value=0.0, max_value=1.0, value=0.65, step=0.01, help="Max vector distance (lower = stricter).")
stream_mode = st.sidebar.toggle("Stream live tokens", value=True)

st.title("RAGChatBot • Streamlit UI")
st.caption("Talk to your ingested docs via FastAPI `/chat`. Toggle streaming on/off in the sidebar.")

# ---------------- Input ----------------
question = st.text_area(
    "Ask a question about your docs",
    placeholder="e.g., What does the README say about the architecture?",
    height=100
)

col1, col2 = st.columns([1, 1])
with col1:
    ask = st.button("Ask", type="primary")
with col2:
    clear = st.button("Clear")

if clear:
    for k in ("last_final", "last_error"):
        st.session_state.pop(k, None)
    st.rerun()

# Placeholders we’ll overwrite (sources only rendered once at the end)
answer_box = st.empty()
sources_box = st.empty()

# ---------------- Helpers ----------------
def parse_sse_lines(lines_iter):
    """
    Minimal SSE parser: yields (event, data_str) tuples.
    Groups consecutive 'data:' lines; event defaults to 'message' if missing.
    """
    event = "message"
    data_parts = []
    for raw in lines_iter:
        if raw is None:
            continue
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        if line == "":
            if data_parts:
                yield event, "\n".join(data_parts)
            event, data_parts = "message", []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            data_parts.append(line.split(":", 1)[1].lstrip())
            continue
    if data_parts:
        yield event, "\n".join(data_parts)

def render_sources(sources):
    sources = sources or []
    with sources_box:
        if not sources:
            st.info("No sources returned.")
            return
        st.subheader("Sources")
        for s in sources:
            idx = s.get("index") or s.get("i") or "?"
            src = s.get("source") or s.get("path") or ""
            filename = s.get("filename") or Path(src).name
            text = (s.get("text") or "").strip()
            snippet = textwrap.shorten(text, width=280, placeholder="…")
            with st.container(border=True):
                st.markdown(f"**[{idx}] {filename}**")
                if src:
                    st.caption(src)
                if snippet:
                    st.write(snippet)

def call_streaming(q: str):
    url = f"{api_base.rstrip('/')}/chat"
    headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["X-API-Key"] = api_key
    params = {"stream": "true", "max_distance": float(max_distance)}
    body = {"question": q, "top_k": int(top_k)}

    try:
        with requests.post(url, params=params, json=body, headers=headers, timeout=300, stream=True) as resp:
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                return None, f"Error {resp.status_code}: {detail}"

            running_answer = ""
            seen_sources = []

            for event, data_str in parse_sse_lines(resp.iter_lines(decode_unicode=True)):
                if not data_str:
                    continue
                try:
                    payload = json.loads(data_str)
                except Exception:
                    payload = {"content": data_str}

                name = payload.get("event") or event

                if name == "meta":
                    # capture sources but DO NOT render yet
                    seen_sources = payload.get("sources") or seen_sources

                elif name == "token":
                    token = payload.get("content") or payload.get("token") or ""
                    running_answer += token
                    answer_box.markdown(running_answer)

                elif name == "final":
                    # normalize final payload; still do NOT render sources yet
                    data = payload.get("data", payload)
                    if isinstance(data, dict):
                        running_answer = data.get("answer") or running_answer
                        seen_sources = data.get("sources") or seen_sources
                    elif isinstance(data, str):
                        running_answer = data

                    answer_box.markdown(running_answer)
                    st.session_state["last_final"] = {"answer": running_answer, "sources": seen_sources}

                elif name == "error":
                    msg = payload.get("message") or data_str
                    st.session_state["last_error"] = msg
                    return None, msg

                elif name == "done":
                    # return whatever we have; sources will be rendered once below
                    return st.session_state.get("last_final") or {"answer": running_answer, "sources": seen_sources}, None

                else:
                    # Unknown event -> treat as token if it has content
                    token = payload.get("content") or ""
                    if token:
                        running_answer += token
                        answer_box.markdown(running_answer)

            # Stream ended w/o explicit "done"
            return st.session_state.get("last_final") or {"answer": running_answer, "sources": seen_sources}, None
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"

def call_non_streaming(q: str):
    url = f"{api_base.rstrip('/')}/chat"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    params = {"stream": "false", "max_distance": float(max_distance)}
    body = {"question": q, "top_k": int(top_k)}
    try:
        resp = requests.post(url, params=params, json=body, headers=headers, timeout=120)
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            return None, f"Error {resp.status_code}: {detail}"
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"

# ---------------- Action ----------------
if ask and question.strip():
    # Clear on each ask
    answer_box.markdown("")
    sources_box.empty()

    with st.spinner("Contacting API..."):
        if stream_mode:
            data, err = call_streaming(question.strip())
        else:
            data, err = call_non_streaming(question.strip())

    if err:
        st.error(err)
        st.session_state["last_error"] = err
    else:
        st.session_state["last_final"] = data
        st.session_state.pop("last_error", None)

# ---------------- Final state render (once) ----------------
final = st.session_state.get("last_final")
if final:
    if final.get("answer"):
        answer_box.markdown(final["answer"])
    if final.get("sources"):
        render_sources(final["sources"])

error = st.session_state.get("last_error")
if error:
    st.error(error)

st.markdown("---")
st.caption("Sources are rendered once after completion (no mid-stream duplicates). Ensure the **X-API-Key** matches your FastAPI server.")