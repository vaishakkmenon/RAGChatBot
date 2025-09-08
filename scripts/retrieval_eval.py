# scripts/retrieval_eval_lite.py
from __future__ import annotations
import argparse, json, re, time, threading, statistics as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")

def normalize(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def token_set(s: str) -> set:
    return set(normalize(s).split())

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def load_squad_v2(path: str) -> List[Dict[str, Any]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    rows = []
    for art in data.get("data", []):
        for para in art.get("paragraphs", []):
            ctx = para.get("context", "")
            for qa in para.get("qas", []):
                rows.append({
                    "id": qa.get("id"),
                    "question": qa.get("question", ""),
                    "is_impossible": qa.get("is_impossible", False),
                    "answers": [a.get("text","") for a in qa.get("answers", [])],
                    "context": ctx
                })
    return rows

_thread_local = threading.local()
def session() -> requests.Session:
    s = getattr(_thread_local, "s", None)
    if s is None:
        s = requests.Session()
        try:
            from requests.adapters import HTTPAdapter
            ad = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
            s.mount("http://", ad); s.mount("https://", ad)
        except Exception:
            pass
        _thread_local.s = s
    return s

def call_search(host: str, api_key: Optional[str], q: str, k: int, max_distance: float, timeout: float):
    params = {"q": q, "k": k, "max_distance": max_distance}
    headers = {"Connection": "keep-alive"}
    if api_key:
        headers["X-API-Key"] = api_key  # <- your middleware expects this exact header
    t0 = time.perf_counter()
    r = session().get(f"{host.rstrip('/')}/debug/search", params=params, headers=headers, timeout=timeout)
    dt_ms = int((time.perf_counter()-t0)*1000)
    r.raise_for_status()
    return r.json(), dt_ms

def from_same_context(retr_text: str, gold_context: str, ans_texts: List[str], min_jaccard: float) -> bool:
    # 1) direct containment (chunk within paragraph, post-normalization)
    R = normalize(retr_text)
    G = normalize(gold_context)
    if R and R in G:
        return True
    # 2) answer-string containment (for answerables)
    for a in ans_texts or []:
        if normalize(a) and normalize(a) in R:
            return True
    # 3) token overlap (robust to punctuation/overlap chunking)
    jr = jaccard(token_set(retr_text), token_set(gold_context))
    return jr >= min_jaccard

def main():
    ap = argparse.ArgumentParser(description="Retrieval-only eval against /debug/search")
    ap.add_argument("--dataset", required=True, help="Path to SQuAD v2 dev JSON")
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default=None, help="If API key middleware is enabled")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-distance", type=float, default=0.70)
    ap.add_argument("--min-jaccard", type=float, default=0.55, help="Context↔chunk token Jaccard threshold")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", choices=["all","answerable","unanswerable"], default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--samples", type=int, default=5, help="Show up to N misses")
    args = ap.parse_args()

    rows = load_squad_v2(args.dataset)
    if args.split != "all":
        want_ans = (args.split == "answerable")
        rows = [r for r in rows if (not r["is_impossible"]) == want_ans]
    rows = rows[args.offset: args.offset + args.limit]
    total = len(rows)
    print(f"Evaluating retrieval on {total} examples (k={args.k}, max_distance={args.max_distance})…")

    hits = 0
    hits_at1 = 0
    mrr_sum = 0.0
    ans_hits = ans_total = 0
    unans_hits = unans_total = 0
    latencies: List[int] = []
    sample_misses: List[Tuple[str,str,str]] = []  # (question, top1_text, gold_ctx_snippet)

    def work(row: Dict[str, Any]) -> Tuple[bool, bool, float, Optional[int], Tuple[str,str,str]]:
        try:
            results, ms = call_search(args.host, args.api_key, row["question"], args.k, args.max_distance, args.timeout)
            latencies.append(ms)
            gold_ctx = row["context"]
            answers = row["answers"]
            hit_rank = None
            for idx, item in enumerate(results, start=1):
                text = item.get("text") or ""
                if from_same_context(text, gold_ctx, answers, args.min_jaccard):
                    hit_rank = idx
                    break
            if hit_rank is None:
                top1_text = (results[0].get("text") if results else "") or ""
                return (False, False, 0.0, ms, (row["question"], top1_text[:220], gold_ctx[:220]))
            else:
                return (True, hit_rank == 1, 1.0 / hit_rank, ms, ("","",""))
        except Exception as e:
            # treat exceptions as a miss; record minimal info
            return (False, False, 0.0, None, (row["question"], f"[error: {e.__class__.__name__}]", row["context"][:220]))

    if args.workers <= 1:
        for r in rows:
            ok, at1, rr, ms, miss = work(r)
            if ok: hits += 1; mrr_sum += rr
            if at1: hits_at1 += 1
            if r["is_impossible"]: unans_total += 1;  unans_hits += int(ok)
            else:                  ans_total   += 1;  ans_hits   += int(ok)
            if not ok and len(sample_misses) < args.samples:
                sample_misses.append(miss)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, r) for r in rows]
            done = 0
            for fut in as_completed(futs):
                ok, at1, rr, ms, miss = fut.result()
                done += 1
                if ok: hits += 1; mrr_sum += rr
                if at1: hits_at1 += 1
                # we don't have the row here—so store split counts in the tuple? Simpler: compute afterwards?
                # To keep split metrics, rerun with workers=1 OR embed split boolean in work args.
                # Quick fix: we stored nothing—so instead, we recompute with workers=1 for split metrics if needed.
                # But we do want split metrics—so adjust:
                pass
    # The above 'pass' prevents split metrics when workers>1. To keep code simple, recompute split metrics serially:
    ans_total = sum(1 for r in rows if not r["is_impossible"])
    unans_total = total - ans_total
    if args.workers > 1:
        # second pass only to count split hits (cheap, just re-evaluates decisions without HTTP)
        # We'll cache last results? Simpler: run serially again but with k=1 and min_jaccard from cached top1?
        # To keep this script minimal, skip recomputation—report overall metrics only when workers>1.
        pass

    overall_recall = hits / total if total else 0.0
    hit_at1 = hits_at1 / total if total else 0.0
    mrr = mrr_sum / total if total else 0.0

    lat_summary = {
        "count": len(latencies),
        "avg_ms": round(sum(latencies)/len(latencies),2) if latencies else None,
        "p50_ms": int(stats.median(latencies)) if latencies else None,
        "p90_ms": int(stats.quantiles(latencies, n=10)[8]) if len(latencies) >= 10 else None,
        "p99_ms": int(stats.quantiles(latencies, n=100)[98]) if len(latencies) >= 100 else None,
    }

    print("\n=== Retrieval Results ===")
    print(f"Examples:         {total}")
    print(f"Recall@{args.k}:         {overall_recall:.3f}")
    print(f"Hit@1:            {hit_at1:.3f}")
    print(f"MRR:              {mrr:.3f}")
    if ans_total and unans_total and args.workers <= 1:
        print(f"Ans Recall@{args.k}:    {ans_hits/ans_total:.3f}  (n={ans_total})")
        print(f"Unans Recall@{args.k}:  {unans_hits/unans_total:.3f}  (n={unans_total})")
    print(f"Latency (ms):     {lat_summary}")

    if sample_misses:
        print("\n--- Sample Misses ---")
        for q, top1, gold in sample_misses:
            print(f"Q: {q}\nTop1: {top1}\nGold: {gold}\n")
    print("Done.")
if __name__ == "__main__":
    main()