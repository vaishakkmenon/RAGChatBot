#!/usr/bin/env python3
"""
squad_search.py — unified grid/random search driver for your RAG QA thresholds

This tool supersedes `squad_grid.py` and adds a RANDOM mode while remaining fully
compatible with your existing GRID workflow. It shells out to `squad_eval.py` for
scoring and writes an append-only CSV (with resume support).

USAGE OVERVIEW
--------------

# 1) GRID mode (replicates your current behavior)
docker compose exec api python /workspace/scripts/squad_search.py ^
  --mode grid ^
  --dataset /workspace/data/squad/dev-v2.0.json ^
  --api-key my-dev-key-1 ^
  --limit 200 --timeout 300 ^
  --k-list 3 4 5 ^
  --null-list 0.60 0.65 0.70 ^
  --maxdist-list 0.60 0.65 ^
  --extractive ^
  --alpha-list 0.50 ^
  --hits-list 2 ^
  --support-list 0.30 0.35 0.40 ^
  --span-list 0.45 0.50 0.55 0.60 1.00 ^
  --support-window 128 ^
  --out /workspace/search_grid.csv --resume

# 2) RANDOM mode (recommended for wider exploration)
docker compose exec api python /workspace/scripts/squad_search.py ^
  --mode random ^
  --dataset /workspace/data/squad/dev-v2.0.json ^
  --api-key my-dev-key-1 ^
  --limit 200 --timeout 300 ^
  --n-samples 300 --seed 42 ^
  --k-min 3 --k-max 8 ^
  --null-min 0.55 --null-max 0.85 ^
  --maxd-min 0.45 --maxd-max 0.80 ^
  --span-min 0.45 --span-max 0.80 --span-jitter 0.05 ^
  --support-min-min 0.30 --support-min-max 0.60 ^
  --support-window-choices 64 96 128 160 192 224 256 ^
  --alpha-min 0.30 --alpha-max 0.70 --alpha-hits-choices 1 2 3 ^
  --extractive ^
  --out /workspace/search_random.csv --resume

NOTES
-----
* Both modes respect --resume: already-seen (param, dataset, host, limit) tuples
  are skipped from the output CSV.
* RANDOM mode uses uniform sampling by default and lightly correlates
  `span_max_distance` with `max_distance` via ±span-jitter to avoid a looser
  span gate than retrieval.
* This script prints a compact leaderboard at the end (Top 10 by F1 → NoAns → AnsF1).

"""

import argparse, csv, json, os, subprocess, sys, random
from datetime import datetime

# NEW: tqdm progress bar
from tqdm import tqdm

# ------------------ shared plumbing (aligned with your evaluator) ------------------

def _extract_last_json_blob(s: str) -> str:
    start = None
    depth = 0
    last = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last = s[start:i+1]
    if last is None:
        raise ValueError("No JSON object found in output")
    return last

def run_eval(dataset, api_key, k, null_thr, max_dist, limit, timeout,
             extractive=False, alpha=None, alpha_hits=None,
             support_min=None, support_window=None, span_max_distance=None,
             eval_workers=1, host="http://127.0.0.1:8000"):
    cmd = [
        "python", "/workspace/scripts/squad_eval.py",
        "--dataset", dataset,
        "--host", host,
        "--api-key", api_key,
        "--grounded-only", "--temperature", "0",
        "--top-k", str(k),
        "--max-distance", f"{max_dist:.2f}",
        "--null-threshold", f"{null_thr:.2f}",
        "--workers", str(eval_workers),
        "--timeout", str(timeout),
        "--limit", str(limit),
    ]
    if extractive:
        cmd.append("--extractive")
        if alpha is not None:
            cmd += ["--alpha", f"{alpha:.2f}"]
        if alpha_hits is not None:
            cmd += ["--alpha-hits", str(int(alpha_hits))]
        if support_min is not None:
            cmd += ["--support-min", f"{support_min:.2f}"]
        if support_window is not None:
            cmd += ["--support-window", str(int(support_window))]
        if span_max_distance is not None:
            cmd += ["--span-max-distance", f"{span_max_distance:.2f}"]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    try:
        blob = _extract_last_json_blob(out)
        return json.loads(blob), out
    except Exception:
        print("=== evaluator output (tail) ===")
        print(out[-2000:])
        raise

HDR = [
    "status","mode","k","null_threshold","max_distance",
    "extractive","alpha","alpha_hits","support_min","support_window","span_max_distance",
    "n","EM","F1","ans_n","ans_EM","ans_F1","un_n","noans_acc","avg_ms","p50_ms",
    "limit","timeout","dataset","host"
]

def ensure_header(path):
    need_header = True
    if os.path.exists(path):
        try:
            need_header = os.path.getsize(path) == 0
        except Exception:
            need_header = True
    if need_header:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HDR)

def append_row(path, row_dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row_dict.get(h, "") for h in HDR])

def load_done_keys(path):
    """
    Build a resume set covering BOTH generative and extractive knobs,
    so interrupted runs can be resumed safely. Keys include mode to avoid
    collisions when you reuse the same CSV for grid and random.
    """
    done = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return done
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            try:
                mode = d.get("mode") or "grid"
                k = int(d["k"])
                null_thr = round(float(d["null_threshold"]), 4)
                maxd = round(float(d["max_distance"]), 4)
                extractive = str(d.get("extractive","")).strip()
                ex_flag = (extractive in ("1","true","True","YES","yes"))
                alpha = d.get("alpha"); alpha = None if alpha in ("", None) else round(float(alpha),4)
                hits = d.get("alpha_hits"); hits = None if hits in ("", None) else int(hits)
                supp = d.get("support_min"); supp = None if supp in ("", None) else round(float(supp),4)
                swin = d.get("support_window"); swin = None if swin in ("", None) else int(swin)
                sdist = d.get("span_max_distance"); sdist = None if sdist in ("", None) else round(float(sdist),4)
                limit = int(d.get("limit", "0") or 0)
                dataset = d.get("dataset") or ""
                host = d.get("host") or "http://127.0.0.1:8000"
                done.add((mode, k, null_thr, maxd, ex_flag, alpha, hits, supp, swin, sdist, limit, dataset, host))
            except Exception:
                continue
    return done

# ------------------ GRID enumerator (keeps your original behavior) ------------------

def combos_from_lists(k_list, null_list, maxdist_list,
                      extractive, alpha_list, hits_list, support_list, span_list,
                      support_window, host):
    out = []
    for k in k_list:
        for nt in null_list:
            for md in maxdist_list:
                if not extractive:
                    out.append((int(k), round(float(nt),4), round(float(md),4),
                                False, None, None, None, support_window, None, host))
                else:
                    for a in alpha_list:
                        for h in hits_list:
                            for sm in support_list:
                                for sd in span_list:
                                    out.append((int(k), round(float(nt),4), round(float(md),4),
                                                True, round(float(a),4), int(h),
                                                round(float(sm),4), support_window,
                                                round(float(sd),4), host))
    return out

# ------------------ RANDOM sampler ------------------

def sample_random(args):
    """
    Yield N random parameter tuples. We correlate span_max_distance with max_distance
    using ±span_jitter and clamp to [span_min, span_max].
    """
    rnd = random.Random(args.seed)
    windows = args.support_window_choices or [128]
    alpha_hits_choices = args.alpha_hits_choices or [2]

    for _ in range(args.n_samples):
        k = rnd.randint(args.k_min, args.k_max)
        nt = rnd.uniform(args.null_min, args.null_max)
        md = rnd.uniform(args.maxd_min, args.maxd_max)

        # span gate around md with jitter
        base_span = md
        if args.span_min is not None and args.span_max is not None:
            jitter = rnd.uniform(-args.span_jitter, args.span_jitter)
            span = max(args.span_min, min(args.span_max, base_span + jitter))
        else:
            span = md  # fallback

        sm = rnd.uniform(args.support_min_min, args.support_min_max)
        sw = rnd.choice(windows)

        if args.extractive:
            alpha = rnd.uniform(args.alpha_min, args.alpha_max) if args.alpha_min is not None else None
            hits = rnd.choice(alpha_hits_choices)
            ex_flag = True
        else:
            alpha = None
            hits = None
            ex_flag = False

        yield (k, round(nt,4), round(md,4), ex_flag,
               (round(alpha,4) if alpha is not None else None),
               (int(hits) if hits is not None else None),
               round(sm,4), int(sw), round(span,4), args.host)

# ------------------ resume hint (grid-like for convenience) ------------------

def build_resume_hint(args):
    base = [
        'docker compose exec api python /workspace/scripts/squad_search.py ^',
        f'  --mode {args.mode} ^',
        f'  --dataset {args.dataset} ^',
        f'  --api-key {args.api_key} ^',
        f'  --limit {args.limit} --timeout {args.timeout} ^',
        f'  --host {args.host} ^',
        f'  --eval-workers {args.eval_workers} ^',
    ]
    if args.mode == "grid":
        base += [
            f'  --k-list {" ".join(map(str, args.k_list))} ^',
            f'  --null-list {" ".join(map(str, args.null_list))} ^',
            f'  --maxdist-list {" ".join(map(str, args.maxdist_list))} ^',
        ]
        if args.extractive:
            base += [
                '  --extractive ^',
                f'  --alpha-list {" ".join(map(str, args.alpha_list))} ^',
                f'  --hits-list {" ".join(map(str, args.hits_list))} ^',
                f'  --support-list {" ".join(map(str, args.support_list))} ^',
                f'  --span-list {" ".join(map(str, args.span_list))} ^',
                f'  --support-window {args.support_window} ^',
            ]
    else:  # random
        base += [
            f'  --n-samples {args.n_samples} --seed {args.seed} ^',
            f'  --k-min {args.k_min} --k-max {args.k_max} ^',
            f'  --null-min {args.null_min:.2f} --null-max {args.null_max:.2f} ^',
            f'  --maxd-min {args.maxd_min:.2f} --maxd-max {args.maxd_max:.2f} ^',
            f'  --span-min {args.span_min:.2f} --span-max {args.span_max:.2f} --span-jitter {args.span_jitter:.2f} ^',
            f'  --support-min-min {args.support_min_min:.2f} --support-min-max {args.support_min_max:.2f} ^',
            f'  --support-window-choices {" ".join(map(str, args.support_window_choices))} ^',
        ]
        if args.extractive:
            base += [
                '  --extractive ^',
                f'  --alpha-min {args.alpha_min:.2f} --alpha-max {args.alpha_max:.2f} ^',
                f'  --alpha-hits-choices {" ".join(map(str, args.alpha_hits_choices))} ^',
            ]
    base.append(f'  --out {args.out} --resume')
    return "\n".join(base)

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Unified GRID/RANDOM search over QA thresholds (calls squad_eval.py)")
    ap.add_argument("--mode", choices=["grid","random"], default="grid")

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--host", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--eval-workers", type=int, default=1,
                help="Number of parallel workers for squad_eval.py")

    # GRID params (compatible with old script)
    ap.add_argument("--k-list", nargs="+", type=int, default=[3,4,5])
    ap.add_argument("--null-list", nargs="+", type=float, default=[0.60,0.65,0.70])
    ap.add_argument("--maxdist-list", nargs="+", type=float, default=[0.60,0.65])

    ap.add_argument("--extractive", action="store_true")
    ap.add_argument("--alpha-list", nargs="+", type=float, default=[0.50])
    ap.add_argument("--hits-list", nargs="+", type=int, default=[2])
    ap.add_argument("--support-list", nargs="+", type=float, default=[0.30,0.35,0.40])
    ap.add_argument("--span-list", nargs="+", type=float, default=[0.45,0.50,0.55,0.60,1.00])
    ap.add_argument("--support-window", type=int, default=128)

    # RANDOM params
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--null-min", type=float, default=0.55)
    ap.add_argument("--null-max", type=float, default=0.85)
    ap.add_argument("--maxd-min", type=float, default=0.45)
    ap.add_argument("--maxd-max", type=float, default=0.80)

    ap.add_argument("--span-min", type=float, default=0.45)
    ap.add_argument("--span-max", type=float, default=0.80)
    ap.add_argument("--span-jitter", type=float, default=0.05)

    ap.add_argument("--support-min-min", type=float, default=0.30)
    ap.add_argument("--support-min-max", type=float, default=0.60)
    ap.add_argument("--support-window-choices", nargs="+", type=int, default=[64,96,128,160,192,224,256])

    ap.add_argument("--alpha-min", type=float, default=0.30)
    ap.add_argument("--alpha-max", type=float, default=0.70)
    ap.add_argument("--alpha-hits-choices", nargs="+", type=int, default=[1,2,3])

    # Legacy combos (generative-only)
    ap.add_argument("--combos", type=str, default=None, help="Semicolon-separated k:null:maxd list (generative-only)")

    ap.add_argument("--out", default=f"/workspace/squad_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    ap.add_argument("--resume", action="store_true", help="Skip items already present in --out")
    args = ap.parse_args()

    # Determine enumeration strategy
    if args.mode == "grid":
        if args.combos:
            triples = []
            for part in (args.combos or "").split(";"):
                part = part.strip()
                if not part: continue
                try:
                    k_s, nt_s, md_s = part.split(":")
                    triples.append((int(k_s), round(float(nt_s),4), round(float(md_s),4),
                                    False, None, None, None, None, None, args.host))
                except Exception:
                    raise SystemExit(f"Bad --combos entry: {part} (expected k:null:maxd;...)")
        else:
            triples = combos_from_lists(
                args.k_list, args.null_list, args.maxdist_list,
                args.extractive, args.alpha_list, args.hits_list,
                args.support_list, args.span_list, args.support_window, args.host
            )
    else:  # random
        triples = list(sample_random(args))

    ensure_header(args.out)
    done = load_done_keys(args.out) if args.resume else set()

    # Filter remaining
    remaining = []
    for (k, nt, md, ex, a, h, sm, sw, sd, host) in triples:
        key = (args.mode, k, nt, md, ex, a, h, sm, sw, sd, args.limit, args.dataset, host)
        if key not in done:
            remaining.append((k, nt, md, ex, a, h, sm, sw, sd, host))

    total = len(remaining)
    if total == 0:
        print("Nothing to do: all items already present in CSV (consider changing lists/ranges or output path).")
        return

    # tqdm progress bar
    pbar = tqdm(total=total, ncols=100, desc=f"{args.mode.upper()} search", unit="run")

    try:
        for idx, (k, nt, md, ex, a, h, sm, sw, sd, host) in enumerate(remaining, 1):
            tag = (
                f"k={k} null={nt:.2f} maxd={md:.2f}"
                + (f" EX a={a} h={h} supp={sm} span={sd}" if ex else "")
            )
            pbar.set_postfix_str(tag, refresh=False)

            status = "ok"
            try:
                j, raw = run_eval(
                    args.dataset, args.api_key, k, nt, md, args.limit, args.timeout,
                    extractive=ex, alpha=a, alpha_hits=h,
                    support_min=sm, support_window=sw, span_max_distance=sd, 
                    eval_workers=args.eval_workers, host=host
                )
                over = j.get("overall", {})
                ans  = j.get("answerable", {})
                un   = j.get("unanswerable", {})
                lat  = j.get("latency", {})

                # Show measured latency in the bar if available
                ms = lat.get("avg_ms") or lat.get("p50_ms")
                if ms:
                    pbar.set_postfix_str(f"{tag} | ~{float(ms):.0f}ms", refresh=False)

                row = {
                    "status": status,
                    "mode": args.mode,
                    "k": k, "null_threshold": nt, "max_distance": md,
                    "extractive": int(ex), "alpha": a, "alpha_hits": h,
                    "support_min": sm, "support_window": sw, "span_max_distance": sd,
                    "n": j.get("n"),
                    "EM": over.get("EM"), "F1": over.get("F1"),
                    "ans_n": ans.get("n"), "ans_EM": ans.get("EM"), "ans_F1": ans.get("F1"),
                    "un_n": un.get("n"), "noans_acc": un.get("NoAns_Accuracy"),
                    "avg_ms": lat.get("avg_ms"), "p50_ms": lat.get("p50_ms"),
                    "limit": args.limit, "timeout": args.timeout, "dataset": args.dataset, "host": host,
                }
                append_row(args.out, row)
            except KeyboardInterrupt:
                pbar.close()
                print("\nInterrupted by user.")
                raise
            except Exception as e:
                status = "error"
                row = {
                    "status": status,
                    "mode": args.mode,
                    "k": k, "null_threshold": nt, "max_distance": md,
                    "extractive": int(ex), "alpha": a, "alpha_hits": h,
                    "support_min": sm, "support_window": sw, "span_max_distance": sd,
                    "n": "", "EM": "", "F1": "",
                    "ans_n": "", "ans_EM": "", "ans_F1": "",
                    "un_n": "", "noans_acc": "",
                    "avg_ms": "", "p50_ms": "",
                    "limit": args.limit, "timeout": args.timeout, "dataset": args.dataset, "host": host,
                }
                append_row(args.out, row)
                pbar.set_postfix_str(f"{tag} | ERROR", refresh=False)
                pbar.write(f"[WARN] Failed: {tag}. Error recorded and continuing. ({e})")
                # continue regardless
            finally:
                pbar.update(1)
    except KeyboardInterrupt:
        pbar.close()
        print("\nTo resume ONLY what remains, re-run the same command with --resume, e.g.:")
        print(build_resume_hint(args))
        return
    finally:
        pbar.close()

    # Leaderboard (Top 10 by F1 → NoAns → AnsF1)
    print(f"\n{args.mode.upper()} complete. CSV saved to: {args.out}")
    try:
        rows = []
        with open(args.out, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for d in r:
                if d.get("status") != "ok":
                    continue
                try:
                    rows.append({
                        "k": int(d["k"]),
                        "null": float(d["null_threshold"]),
                        "maxd": float(d["max_distance"]),
                        "extractive": d.get("extractive",""),
                        "alpha": d.get("alpha",""),
                        "hits": d.get("alpha_hits",""),
                        "support_min": d.get("support_min",""),
                        "span_max_distance": d.get("span_max_distance",""),
                        "F1": float(d["F1"] or 0),
                        "EM": float(d["EM"] or 0),
                        "noans": float(d["noans_acc"] or 0),
                        "ansF1": float(d.get("ans_F1") or 0),
                    })
                except Exception:
                    pass
        rows.sort(key=lambda d: (-d["F1"], -d["noans"], -d["ansF1"]))
        print("\nTop 10 (Overall F1, tie → NoAns, then Answerable F1):")
        for i, d in enumerate(rows[:10], 1):
            ex = "EX" if str(d.get("extractive","")).lower() in ("1","true","yes") else "GEN"
            print(f" {i:>2}. [{ex}] F1={d['F1']:.3f} noans={d['noans']:.3f} ansF1={d['ansF1']:.3f}  "
                  f"k={d['k']} null={d['null']:.2f} maxd={d['maxd']:.2f} "
                  f"alpha={d.get('alpha','')} hits={d.get('hits','')} "
                  f"supp={d.get('support_min','')} span={d.get('span_max_distance','')}")
    except Exception as e:
        print(f"(skip leaderboard) {e}")

if __name__ == "__main__":
    main()