#!/usr/bin/env python3
# scripts/squad_eval.py
from __future__ import annotations
import argparse, json, re, time, threading, statistics as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional

import requests

# -------- Normalization / Metrics (your original style, kept) --------
_ARTICLES = {"a","an","the"}
_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~]")
_WS_RE = re.compile(r"\s+")
IDK_RE = re.compile(
    r"\b("
    r"i\s*(?:do\s*not|don['’]t)\s*know"
    r"|cannot\s+answer|can['’]t\s+answer"
    r"|no\s+answer|not\s+sure|unsure|unknown|no\s+information"
    r"|information\s+not\s+available|insufficient\s+information"
    r"|answer\s+not\s+found|the\s+context\s+does\s+not\s+contain"
    r"|not\s+in\s+context"
    r")\b",
    re.I,
)

def _normalize(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = " ".join(w for w in s.split() if w not in _ARTICLES)
    return _WS_RE.sub(" ", s).strip()

def exact_match(pred: str, golds: List[str]) -> int:
    p = _normalize(pred)
    return int(any(p == _normalize(g) for g in golds))

def f1_score(pred: str, golds: List[str]) -> float:
    def toks(s: str) -> List[str]: return _normalize(s).split() if s else []
    pt = toks(pred)
    best = 0.0
    for g in golds:
        gt = toks(g)
        if not pt and not gt:
            best = max(best, 1.0); continue
        # multiset overlap
        num_same = 0
        gt_counts: Dict[str,int] = {}
        for w in gt: gt_counts[w] = gt_counts.get(w, 0) + 1
        for w in pt:
            if gt_counts.get(w, 0) > 0:
                num_same += 1
                gt_counts[w] -= 1
        if num_same == 0:
            score = 0.0
        else:
            prec = num_same / len(pt)
            rec  = num_same / len(gt) if gt else 0.0
            score = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        best = max(best, score)
    return best

# -------- Data loading --------
def load_squad_v2(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows: List[Dict[str, Any]] = []
    for art in data.get("data", []):
        for para in art.get("paragraphs", []):
            ctx = para.get("context", "")
            for qa in para.get("qas", []):
                rows.append({
                    "id": qa.get("id"),
                    "question": qa.get("question",""),
                    "is_impossible": qa.get("is_impossible", False),
                    "answers": [a.get("text","") for a in qa.get("answers", [])],
                    "context": ctx,
                })
    return rows

# -------- Request plumbing --------
_thread_local = threading.local()
def session() -> requests.Session:
    s = getattr(_thread_local, "s", None)
    if s is None:
        s = requests.Session()
        _thread_local.s = s
    return s

def call_chat(
    q: str,
    host: str,
    api_key: str,
    grounded_only: bool,
    top_k: int,
    max_distance: float,
    null_threshold: Optional[float],
    temperature: float,
    timeout: float,
    *,
    # Extractive / A2 gates
    extractive: bool = False,
    alpha: Optional[float] = None,
    alpha_hits: Optional[int] = None,
    support_min: Optional[float] = None,
    support_window: Optional[int] = None,
    span_max_distance: Optional[float] = None,
    # Reranker / A3
    rerank: bool = False,
    rerank_lex_w: Optional[float] = None,
    # Sampling
    top_p: Optional[float] = None,
) -> Tuple[str, Optional[int], str]:
    params: Dict[str, Any] = {
        "grounded_only": str(grounded_only).lower(),
        "max_distance": max_distance,
        "temperature": temperature,
    }
    if null_threshold is not None:
        params["null_threshold"] = null_threshold

    # Extractive path
    if extractive:
        params["extractive"] = 1
        if alpha is not None:
            params["alpha"] = alpha
        if alpha_hits is not None:
            params["alpha_hits"] = alpha_hits
        if support_min is not None:
            params["support_min"] = support_min
        if support_window is not None:
            params["support_window"] = support_window
        if span_max_distance is not None:
            params["span_max_distance"] = span_max_distance

    # Reranker
    if rerank:
        params["rerank"] = "true"
        if rerank_lex_w is not None:
            params["rerank_lex_w"] = rerank_lex_w

    # Optional sampling knob
    if top_p is not None:
        params["top_p"] = top_p

    payload = {"question": q, "top_k": top_k}
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "Connection": "keep-alive",
    }
    t0 = time.perf_counter()
    try:
        r = session().post(f"{host.rstrip('/')}/chat", params=params, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        pred = (data.get("answer") or "").strip()
        return pred, int((time.perf_counter()-t0)*1000), "ok"
    except requests.exceptions.RequestException as e:
        return "", None, f"error:{e.__class__.__name__}"

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to SQuAD v2 dev JSON")
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", required=True)

    # Split/slicing
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", choices=["all","answerable","unanswerable"], default="all")

    # Core retrieval / abstention
    ap.add_argument("--grounded-only", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-distance", type=float, default=0.70)
    ap.add_argument("--null-threshold", type=float)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None, help="Optional; server may ignore")

    # Extractive + gates (A2)
    ap.add_argument("--extractive", action="store_true",
                    help="Enable extractive answering (A1+A2) by sending ?extractive=1")
    ap.add_argument("--alpha", type=float, default=None,
                    help="(extractive) distance cutoff for pre-abstain gate (e.g., 0.55)")
    ap.add_argument("--alpha-hits", type=int, default=None,
                    help="(extractive) require at least this many chunks under --alpha")
    ap.add_argument("--support-min", type=float, default=None,
                    help="(extractive) min lexical-overlap support around evidence")
    ap.add_argument("--support-window", type=int, default=None,
                    help="(extractive) half-window chars around evidence for support (e.g., 96/128/160)")
    ap.add_argument("--span-max-distance", type=float, default=None,
                    help="(extractive) evidence chunk distance must be ≤ this value")

    # Reranker (A3)
    ap.add_argument("--rerank", action="store_true", help="Enable reranker")
    ap.add_argument("--rerank-lex-w", type=float, default=0.50, help="Lexical weight when rerank is enabled")

    # Runtime
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", help="Optional path to write JSON summary")

    # Progress (combined system)
    ap.add_argument("--progress", action="store_true", help="Print periodic status updates")
    ap.add_argument("--progress-interval", type=int, default=25, help="Update every N items")

    args = ap.parse_args()

    all_rows = load_squad_v2(args.dataset)
    # Filter slice & split
    rows = all_rows[args.offset: args.offset + args.limit] if args.limit else all_rows[args.offset:]
    if args.split == "answerable":
        rows = [r for r in rows if not r["is_impossible"]]
    elif args.split == "unanswerable":
        rows = [r for r in rows if r["is_impossible"]]

    em_sum = f1_sum = 0.0
    ans_em = ans_f1 = ans_n = 0.0
    unans_em = unans_f1 = unans_n = 0.0
    latencies: List[int] = []
    errors = 0

    def work(row: Dict[str, Any]) -> Tuple[float,float,bool,Optional[int],bool]:
        pred, ms, status = call_chat(
            row["question"], args.host, args.api_key, args.grounded_only,
            args.top_k, args.max_distance, args.null_threshold, args.temperature, args.timeout,
            extractive=args.extractive, alpha=args.alpha, alpha_hits=args.alpha_hits,
            support_min=args.support_min, support_window=args.support_window,
            span_max_distance=args.span_max_distance,
            rerank=args.rerank, rerank_lex_w=args.rerank_lex_w,
            top_p=args.top_p,
        )
        if status != "ok":
            return 0.0, 0.0, row["is_impossible"], None, True
        pred_clean = re.sub(r"\s+"," ", pred).strip()
        pred_eval = "" if IDK_RE.search(pred_clean) else pred_clean
        gts = [""] if row["is_impossible"] else (row["answers"] if row["answers"] else [""])
        EM = float(exact_match(pred_eval, gts))
        F1 = float(f1_score(pred_eval, gts))
        return EM, F1, row["is_impossible"], ms, False

    total = len(rows)
    if args.workers <= 1:
        for i, r in enumerate(rows, 1):
            EM, F1, is_unans, ms, err = work(r)
            if err: errors += 1
            else:
                em_sum += EM; f1_sum += F1
                if is_unans: unans_em += EM; unans_f1 += F1; unans_n += 1
                else:        ans_em += EM;   ans_f1 += F1;   ans_n   += 1
                if ms is not None: latencies.append(ms)
            if args.progress and (i % args.progress_interval == 0 or i == total):
                running_em = em_sum / max(i - errors, 1)
                running_f1 = f1_sum / max(i - errors, 1)
                print(f"  [{i}/{total}] EM={running_em:.3f} F1={running_f1:.3f}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, r) for r in rows]
            done = 0
            for fut in as_completed(futs):
                EM, F1, is_unans, ms, err = fut.result()
                done += 1
                if err: errors += 1
                else:
                    em_sum += EM; f1_sum += F1
                    if is_unans: unans_em += EM; unans_f1 += F1; unans_n += 1
                    else:        ans_em += EM;   ans_f1 += F1;   ans_n   += 1
                    if ms is not None: latencies.append(ms)
                if args.progress and (done % args.progress_interval == 0 or done == total):
                    running_em = em_sum / max(done - errors, 1)
                    running_f1 = f1_sum / max(done - errors, 1)
                    print(f"  [{done}/{total}] EM={running_em:.3f} F1={running_f1:.3f}")

    n = max(total - errors, 1)
    overall_em = em_sum / n
    overall_f1 = f1_sum / n
    ans_em_avg = (ans_em / ans_n) if ans_n else None
    ans_f1_avg = (ans_f1 / ans_n) if ans_n else None
    unans_em_avg = (unans_em / unans_n) if unans_n else None
    unans_f1_avg = (unans_f1 / unans_n) if unans_n else None

    lat_summary = {
        "count": len(latencies),
        "avg_ms": round(sum(latencies)/len(latencies),2) if latencies else None,
        "p50_ms": int(stats.median(latencies)) if latencies else None,
        "p90_ms": int(stats.quantiles(latencies, n=10)[8]) if len(latencies) >= 10 else None,
        "p99_ms": int(stats.quantiles(latencies, n=100)[98]) if len(latencies) >= 100 else None,
    }

    summary = {
        "n": total, "errors": errors,
        "overall": {"EM": round(overall_em,4), "F1": round(overall_f1,4)},
        "answerable": {"n": int(ans_n), "EM": round(ans_em_avg,4) if ans_em_avg is not None else None,
                       "F1": round(ans_f1_avg,4) if ans_f1_avg is not None else None},
        "unanswerable": {"n": int(unans_n), "EM": round(unans_em_avg,4) if unans_em_avg is not None else None,
                         "F1": round(unans_f1_avg,4) if unans_f1_avg is not None else None,
                         "NoAns_Accuracy": round(unans_em_avg,4) if unans_em_avg is not None else None},
        "latency": lat_summary,
        "settings": {
            "host": args.host, "grounded_only": args.grounded_only, "extractive": args.extractive, "top_k": args.top_k,
            "max_distance": args.max_distance, "null_threshold": getattr(args, "null_threshold", None),
            "temperature": args.temperature, "top_p": args.top_p,
            "workers": args.workers, "timeout": args.timeout,
            "alpha": args.alpha, "alpha_hits": args.alpha_hits,
            "support_min": args.support_min, "support_window": args.support_window,
            "span_max_distance": args.span_max_distance,
            "rerank": args.rerank, "rerank_lex_w": args.rerank_lex_w,
        }
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()