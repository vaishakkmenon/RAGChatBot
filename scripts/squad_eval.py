# scripts/squad_eval_lite.py
from __future__ import annotations
import argparse, json, re, time, threading, statistics as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional

import requests

# -------- Normalization / Metrics (kept close to your originals) --------
_ARTICLES = {"a","an","the"}
_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~]")
_WS_RE = re.compile(r"\s+")
IDK_RE = re.compile(
    r"\b("
    r"i\s*(?:do\s*not|don['’]t)\s*know"
    r"|cannot\s+answer|can['’]t\s+answer"
    r"|not\s+in\s+my\s+knowledge\s+base"
    r"|unknown|no\s+answer|n/?a"
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
        common = {}
        for w in pt: 
            if w in gt:
                common[w] = min(pt.count(w), gt.count(w))
        num_same = sum(common.values())
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
    data = json.load(open(json_path, "r", encoding="utf-8"))
    rows = []
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
        try:
            from requests.adapters import HTTPAdapter
            ad = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
            s.mount("http://", ad); s.mount("https://", ad)
        except Exception:
            pass
        _thread_local.s = s
    return s

def call_chat(q: str, host: str, api_key: str, grounded_only: bool, top_k: int,
              max_distance: float, null_threshold: Optional[float], temperature: float,
              timeout: float) -> Tuple[str, Optional[int], str]:
    params = {
        "grounded_only": str(grounded_only).lower(),
        "max_distance": max_distance,
        "temperature": temperature,
    }
    if null_threshold is not None:
        params["null_threshold"] = null_threshold
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
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", choices=["all","answerable","unanswerable"], default="all")
    ap.add_argument("--grounded-only", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-distance", type=float, default=0.70)
    ap.add_argument("--null-threshold", type=float)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", help="Optional path to write JSON summary")
    args = ap.parse_args()

    rows = load_squad_v2(args.dataset)
    if args.split != "all":
        want_ans = (args.split == "answerable")
        rows = [r for r in rows if (not r["is_impossible"]) == want_ans]
    rows = rows[args.offset: args.offset + args.limit]

    print(f"Evaluating {len(rows)} examples…")

    # metrics
    em_sum = f1_sum = 0.0
    ans_em = ans_f1 = ans_n = 0.0
    unans_em = unans_f1 = unans_n = 0.0
    latencies: List[int] = []
    errors = 0

    def work(row: Dict[str, Any]) -> Tuple[float,float,bool,Optional[int],bool]:
        pred, ms, status = call_chat(
            row["question"], args.host, args.api_key, args.grounded_only,
            args.top_k, args.max_distance, args.null_threshold, args.temperature, args.timeout
        )
        if status != "ok":
            return 0.0, 0.0, row["is_impossible"], None, True
        pred_clean = re.sub(r"\s+"," ", pred).strip()
        pred_eval = "" if IDK_RE.search(pred_clean) else pred_clean
        gts = [""] if row["is_impossible"] else (row["answers"] if row["answers"] else [""])
        EM = float(exact_match(pred_eval, gts))
        F1 = float(f1_score(pred_eval, gts))
        return EM, F1, row["is_impossible"], ms, False

    if args.workers <= 1:
        for i, r in enumerate(rows, 1):
            EM, F1, is_unans, ms, err = work(r)
            if err: errors += 1
            else: 
                em_sum += EM; f1_sum += F1
                if is_unans: unans_em += EM; unans_f1 += F1; unans_n += 1
                else:        ans_em += EM;   ans_f1 += F1;   ans_n   += 1
                if ms is not None: latencies.append(ms)
            if i % 25 == 0:
                print(f"  [{i}/{len(rows)}] running EM={em_sum/max(i,1):.3f} F1={f1_sum/max(i,1):.3f}")
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
                if done % 25 == 0:
                    print(f"  [{done}/{len(rows)}] running EM={em_sum/done:.3f} F1={f1_sum/done:.3f}")

    n = max(len(rows) - errors, 1)
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
        "n": len(rows), "errors": errors,
        "overall": {"EM": round(overall_em,4), "F1": round(overall_f1,4)},
        "answerable": {"n": int(ans_n), "EM": round(ans_em_avg,4) if ans_em_avg is not None else None,
                       "F1": round(ans_f1_avg,4) if ans_f1_avg is not None else None},
        "unanswerable": {"n": int(unans_n), "EM": round(unans_em_avg,4) if unans_em_avg is not None else None,
                         "F1": round(unans_f1_avg,4) if unans_f1_avg is not None else None,
                         "NoAns_Accuracy": round(unans_em_avg,4) if unans_em_avg is not None else None},
        "latency": lat_summary,
        "settings": {
            "host": args.host, "grounded_only": args.grounded_only, "top_k": args.top_k,
            "max_distance": args.max_distance, "null_threshold": getattr(args, "null_threshold", None),
            "temperature": args.temperature, "workers": args.workers, "timeout": args.timeout,
        }
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()