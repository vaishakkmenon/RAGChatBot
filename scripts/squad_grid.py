# /workspace/scripts/squad_grid.py
import argparse, csv, json, os, subprocess, sys
from datetime import datetime

def _extract_last_json_blob(s: str) -> str:
    """Return the last balanced JSON object found in s."""
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

def run_eval(dataset, api_key, k, null_thr, max_dist, limit, timeout):
    cmd = [
        "python", "/workspace/scripts/squad_eval.py",
        "--dataset", dataset,
        "--api-key", api_key,
        "--grounded-only", "--temperature", "0",
        "--top-k", str(k),
        "--max-distance", f"{max_dist:.2f}",
        "--null-threshold", f"{null_thr:.2f}",
        "--workers", "1",
        "--timeout", str(timeout),
        "--limit", str(limit),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    try:
        blob = _extract_last_json_blob(out)
        return json.loads(blob), out
    except Exception:
        # Show tail for debugging then re-raise
        print("=== evaluator output (tail) ===")
        print(out[-2000:])
        raise

def fmt(x, nd=3):
    if x is None:
        return ""
    if isinstance(x, (int,)):
        return str(x)
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

HDR = ["status","k","null_threshold","max_distance","n",
       "EM","F1","ans_n","ans_EM","ans_F1","un_n","noans_acc","avg_ms","p50_ms",
       "limit","timeout","dataset"]

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
    done = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return done
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            try:
                k = int(d["k"])
                null_thr = round(float(d["null_threshold"]), 4)
                maxd = round(float(d["max_distance"]), 4)
                limit = int(d.get("limit", "0") or 0)
                dataset = d.get("dataset") or ""
                done.add((k, null_thr, maxd, limit, dataset))
            except Exception:
                continue
    return done

def combos_from_lists(k_list, null_list, maxdist_list):
    out = []
    for k in k_list:
        for nt in null_list:
            for md in maxdist_list:
                out.append((int(k), round(float(nt), 4), round(float(md), 4)))
    return out

def combos_from_arg(s: str):
    # "3:0.60:0.65;5:0.70:0.70"
    out = []
    for part in (s or "").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            k_s, nt_s, md_s = part.split(":")
            out.append((int(k_s), round(float(nt_s),4), round(float(md_s),4)))
        except Exception:
            raise SystemExit(f"Bad --combos entry: {part} (expected k:null:maxd;...)")
    return out

def build_resume_command(args, remaining):
    # Build a CMD command that uses --combos to run only the remaining set
    combo_str = ";".join([f"{k}:{nt:.2f}:{md:.2f}" for (k, nt, md) in remaining])
    cmd = (
        'docker compose exec api python /workspace/scripts/squad_grid.py ^\n'
        f'  --dataset {args.dataset} ^\n'
        f'  --api-key {args.api_key} ^\n'
        f'  --limit {args.limit} --timeout {args.timeout} ^\n'
        f'  --combos {combo_str} ^\n'
        f'  --out {args.out} --resume'
    )
    return cmd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--k-list", nargs="+", type=int, default=[3,5,6])
    ap.add_argument("--null-list", nargs="+", type=float, default=[0.60,0.65,0.70])
    ap.add_argument("--maxdist-list", nargs="+", type=float, default=[0.65,0.70])
    ap.add_argument("--combos", type=str, default=None, help="Semicolon-separated k:null:maxd list")
    ap.add_argument("--out", default=f"/workspace/squad_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    ap.add_argument("--resume", action="store_true", help="Skip combos already present in --out")
    args = ap.parse_args()

    # Determine target combos
    if args.combos:
        triples = combos_from_arg(args.combos)
    else:
        triples = combos_from_lists(args.k_list, args.null_list, args.maxdist_list)

    ensure_header(args.out)
    done = load_done_keys(args.out) if args.resume else set()

    # Filter remaining by subtracting what's already done for this (limit,dataset)
    remaining = []
    for (k, nt, md) in triples:
        key = (k, nt, md, args.limit, args.dataset)
        if key not in done:
            remaining.append((k, nt, md))

    total = len(remaining)
    if total == 0:
        print("Nothing to do: all combos already present in CSV (consider changing lists or limit).")
        return

    print(f"Running grid… ({total} combos)")
    try:
        for idx, (k, nt, md) in enumerate(remaining, 1):
            tag = f"k={k} null={nt:.2f} maxd={md:.2f}"
            print(f"→ [{idx}/{total}] {tag} (limit={args.limit})", flush=True)
            status = "ok"
            try:
                j, raw = run_eval(args.dataset, args.api_key, k, nt, md, args.limit, args.timeout)
                over = j.get("overall", {})
                ans  = j.get("answerable", {})
                un   = j.get("unanswerable", {})
                lat  = j.get("latency", {})

                row = {
                    "status": status,
                    "k": k, "null_threshold": nt, "max_distance": md,
                    "n": j.get("n"),
                    "EM": over.get("EM"), "F1": over.get("F1"),
                    "ans_n": ans.get("n"), "ans_EM": ans.get("EM"), "ans_F1": ans.get("F1"),
                    "un_n": un.get("n"), "noans_acc": un.get("NoAns_Accuracy"),
                    "avg_ms": lat.get("avg_ms"), "p50_ms": lat.get("p50_ms"),
                    "limit": args.limit, "timeout": args.timeout, "dataset": args.dataset,
                }
                append_row(args.out, row)  # ← checkpoint after each combo
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                raise
            except Exception as e:
                status = "error"
                row = {
                    "status": status,
                    "k": k, "null_threshold": nt, "max_distance": md,
                    "n": "", "EM": "", "F1": "",
                    "ans_n": "", "ans_EM": "", "ans_F1": "",
                    "un_n": "", "noans_acc": "",
                    "avg_ms": "", "p50_ms": "",
                    "limit": args.limit, "timeout": args.timeout, "dataset": args.dataset,
                }
                append_row(args.out, row)  # still checkpoint the failure
                print(f"[WARN] Failed: {tag}. Error recorded and continuing. ({e})", file=sys.stderr)
                continue
    except KeyboardInterrupt:
        # Compute what remains and print a ready-to-run CMD command
        done_now = load_done_keys(args.out)
        remain_after_interrupt = []
        for (k, nt, md) in triples:
            if (k, nt, md, args.limit, args.dataset) not in done_now:
                remain_after_interrupt.append((k, nt, md))
        if remain_after_interrupt:
            print("\nTo resume ONLY the remaining combos, run this:")
            print(build_resume_command(args, remain_after_interrupt))
        else:
            print("\nAll combos completed.")
        return

    # Summary
    print(f"\nGrid complete. CSV saved to: {args.out}")
    # Optional: print a top-5 leaderboard by Overall F1 then NoAns then EM
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
                        "F1": float(d["F1"]),
                        "EM": float(d["EM"]),
                        "NoAns": float(d["noans_acc"]) if d.get("noans_acc") else 0.0,
                        "p50": int(d["p50_ms"]) if d.get("p50_ms") else None,
                        "limit": int(d["limit"]),
                    })
                except Exception:
                    continue
        rows = sorted(rows, key=lambda r: (r["F1"], r["NoAns"], r["EM"]), reverse=True)
        print("\nTop results:")
        for r in rows[:5]:
            print(f'k={r["k"]} null={r["null"]:.2f} maxd={r["maxd"]:.2f} '
                  f'F1={fmt(r["F1"])} EM={fmt(r["EM"])} NoAns={fmt(r["NoAns"])} p50={r["p50"]}ms')
    except Exception:
        pass

if __name__ == "__main__":
    main()