
#!/usr/bin/env python
# run_showcase.py
# Runs two highlight SQuADv2 evaluations and builds leaderboards + a Pareto scatter.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def ensure_pkg(mod_name, pip_name=None):
    try:
        __import__(mod_name)
        return True
    except Exception:
        pip_name = pip_name or mod_name
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            __import__(mod_name)
            return True
        except Exception as e:
            print(f"WARNING: Could not install {pip_name}: {e}")
            return False

def find_file(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
        here = Path(__file__).resolve().parent
        p2 = (here / c).resolve()
        if p2.exists():
            return p2
    return None

def run_eval(squad_eval_py, dataset, url, key, out_path, mode):
    # mode: 'gen' (generative grounded) or 'ext' (extractive)
    base = [sys.executable, str(squad_eval_py),
            "--dataset", str(dataset),
            "--host", str(url),
            "--api-key", str(key),
            "--workers", "4", "--timeout", "180",
            "--limit", "500",
            "--progress", "--progress-interval", "50"]
    if mode == "gen":
        args = base + [
            "--grounded-only",
            "--top-k", "5", "--max-distance", "0.60", "--null-threshold", "0.25",
            "--temperature", "0",
            "--out", str(out_path),
            "--rerank", "--rerank-lex-w", "0.5",
        ]
        label = "Generative grounded"
    elif mode == "ext":
        args = base + [
            "--extractive", "--grounded-only",
            "--rerank", "--rerank-lex-w", "0.5",
            "--top-k", "3", "--max-distance", "0.60", "--null-threshold", "0.60",
            "--alpha", "0.50", "--alpha-hits", "2",
            "--support-min", "0.30", "--support-window", "96",
            "--span-max-distance", "0.60",
            "--temperature", "0",
            "--out", str(out_path),
        ]
        label = "Extractive (span-first)"
    else:
        raise ValueError("mode must be 'gen' or 'ext'")

    print(f"\n[{label}] running:")
    print(" ", " ".join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"ERROR: squad_eval.py returned {proc.returncode}")
        sys.exit(proc.returncode)

def build_artifacts_inline(results_folder: Path, out_csv: Path, out_png: Path):
    ok_pd = ensure_pkg("pandas", "pandas")
    ok_plt = ensure_pkg("matplotlib", "matplotlib")
    if not (ok_pd and ok_plt):
        print("ERROR: pandas/matplotlib required to build artifacts.")
        return 1
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []
    for f in sorted(results_folder.glob("*.json")):
        try:
            d = json.load(open(f, "r", encoding="utf-8"))
        except Exception:
            continue
        s = d.get("settings", {})
        o = d.get("overall", {})
        a = d.get("answerable", {})
        u = d.get("unanswerable", {})
        lat = d.get("latency", {})
        rows.append({
            "file": f.name,
            "F1": o.get("F1"),
            "EM": o.get("EM"),
            "AnsF1": a.get("F1"),
            "NoAns": (u or {}).get("NoAns_Accuracy"),
            "p50_ms": (lat or {}).get("p50_ms"),
            "extractive": s.get("extractive"),
            "grounded_only": s.get("grounded_only"),
            "top_k": s.get("top_k"),
            "max_distance": s.get("max_distance"),
            "null_threshold": s.get("null_threshold"),
            "rerank": s.get("rerank"),
            "rerank_lex_w": s.get("rerank_lex_w"),
            "alpha": s.get("alpha"),
            "alpha_hits": s.get("alpha_hits"),
            "support_min": s.get("support_min"),
            "support_window": s.get("support_window"),
            "span_max_distance": s.get("span_max_distance"),
            "temperature": s.get("temperature")
        })
    if not rows:
        print(f"ERROR: No JSON files in {results_folder}")
        return 2
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # Leaderboards (console)
    def show(title, key, asc=False, k=10):
        print("\n=== " + title + " ===")
        cols = ["file","F1","EM","AnsF1","NoAns","p50_ms","extractive","top_k","max_distance","null_threshold","rerank","rerank_lex_w","alpha","alpha_hits","support_min","support_window","span_max_distance"]
        print(df.sort_values(key, ascending=asc).head(k)[cols].to_string(index=False))
    show("Top 10 by Overall F1", "F1", asc=False)
    show("Top 10 by NoAns Accuracy", "NoAns", asc=False)
    show("Top 10 by Answerable F1", "AnsF1", asc=False)

    # Pareto scatter (F1 vs NoAns; size ~ 1/sqrt(p50))
    xs = df["F1"].fillna(0.0).tolist()
    ys = df["NoAns"].fillna(0.0).tolist()
    p50 = df["p50_ms"].fillna(1000).tolist()
    sizes = []
    for v in p50:
        try:
            v = max(1, float(v))
        except Exception:
            v = 1000.0
        sizes.append(2000.0 / (v ** 0.5))
    plt.figure()
    plt.scatter(xs, ys, s=sizes)
    plt.xlabel("Overall F1")
    plt.ylabel("NoAns Accuracy")
    plt.title("Pareto: F1 vs NoAns (size ~ 1/√p50)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_png}")
    return 0

def main():
    ap = argparse.ArgumentParser(description="Run two showcase SQuADv2 evaluations and build leaderboards + plots.")
    ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8000"), help="Service URL")
    ap.add_argument("--key", default=os.environ.get("KEY", "my-dev-key-1"), help="API key header value")
    ap.add_argument("--dataset", default=os.environ.get("DATASET", str(Path.cwd() / "data" / "squad" / "dev-v2.0.json")), help="Path to SQuADv2 dev JSON")
    ap.add_argument("--results", help="Output results folder (default: results/<timestamp>)")
    ap.add_argument("--use-inline-aggregator", action="store_true", help="Ignore build_leaderboards.py and build artifacts inline")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"ERROR: dataset not found: {dataset}")
        sys.exit(2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = Path(args.results) if args.results else Path("results") / ts
    results.mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(parents=True, exist_ok=True)

    squad_eval_py = find_file(["scripts/squad_eval.py", "scripts\\squad_eval.py"])
    if not squad_eval_py:
        print("ERROR: scripts/squad_eval.py not found. Please ensure it exists.")
        sys.exit(3)

    print("\n=== Using ===")
    print("URL=", args.url)
    print("KEY=", args.key)
    print("DATASET=", dataset)
    print("RESULTS=", results)

    run_eval(squad_eval_py, dataset, args.url, args.key, results / "gen_k5_md060_nt0p25_rerank1.json", "gen")
    run_eval(squad_eval_py, dataset, args.url, args.key, results / "ex_sd0p60.json", "ext")

    if not args.use_inline_aggregator:
        build_py = find_file(["scripts/build_leaderboards.py", "scripts\\build_leaderboards.py"])
        if build_py:
            print("\n[Aggregator] Using scripts/build_leaderboards.py")
            cmd = [sys.executable, str(build_py), "--folder", str(results)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(proc.stdout)
            if proc.returncode == 0:
                print("Artifacts built via build_leaderboards.py")
                return
            else:
                print("WARNING: build_leaderboards.py failed; falling back to inline aggregator.")

    code = build_artifacts_inline(results, Path("docs") / "combined_eval_table.csv", Path("docs") / "pareto_scatter.png")
    if code != 0:
        sys.exit(code)

if __name__ == "__main__":
    main()
