#!/usr/bin/env python
# run_grid_showcase.py
# Runs a small but diverse grid of SQuADv2 evaluations and aggregates results.
# Produces docs/combined_eval_table.csv and docs/pareto_scatter.png.

import argparse, json, os, subprocess, sys
from datetime import datetime
from pathlib import Path

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

def run_eval(squad_eval_py, cfg, common, out_path):
    args = [sys.executable, str(squad_eval_py),
            "--dataset", str(common["dataset"]),
            "--host", str(common["url"]),
            "--api-key", str(common["key"]),
            "--workers", str(common["workers"]),
            "--timeout", str(common["timeout"]),
            "--limit", str(common["limit"]),
            "--progress", "--progress-interval", "50",
            "--out", str(out_path)]
    # mode flags
    if cfg.get("extractive"):
        args += ["--extractive"]
    if cfg.get("grounded_only", True):
        args += ["--grounded-only"]

    # common knobs
    args += ["--top-k", str(cfg["top_k"])]
    args += ["--max-distance", f"{cfg.get('max_distance', 0.60):.2f}"]
    if cfg.get("null_threshold") is not None:
        args += ["--null-threshold", f"{cfg['null_threshold']:.2f}"]
    args += ["--temperature", str(cfg.get("temperature", 0))]

    # reranker
    if cfg.get("rerank", False):
        args += ["--rerank", "--rerank-lex-w", str(cfg.get("rerank_lex_w", 0.5))]

    # extractive gates
    if cfg.get("extractive"):
        if cfg.get("alpha") is not None:
            args += ["--alpha", f"{cfg['alpha']:.2f}"]
        if cfg.get("alpha_hits") is not None:
            args += ["--alpha-hits", str(cfg["alpha_hits"])]
        if cfg.get("support_min") is not None:
            args += ["--support-min", f"{cfg['support_min']:.2f}"]
        if cfg.get("support_window") is not None:
            args += ["--support-window", str(cfg["support_window"])]
        if cfg.get("span_max_distance") is not None:
            args += ["--span-max-distance", f"{cfg['span_max_distance']:.2f}"]

    print("\n[RUN]", out_path.name)
    print(" ", " ".join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"ERROR: squad_eval.py returned {proc.returncode} for {out_path.name}")
        return False
    return True

def ensure_pkg(mod_name, pip_name=None):
    try:
        __import__(mod_name); return True
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or mod_name])
            __import__(mod_name); return True
        except Exception as e:
            print(f"WARNING: could not install {pip_name or mod_name}: {e}")
            return False

def aggregate(results_folder: Path, out_csv: Path, out_png: Path):
    ok_pd = ensure_pkg("pandas", "pandas")
    ok_plt = ensure_pkg("matplotlib", "matplotlib")
    if not (ok_pd and ok_plt):
        print("ERROR: pandas/matplotlib are required to aggregate.")
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
        print(f"ERROR: no JSON results in {results_folder}")
        return 2
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # Pareto frontier indices (maximize F1 and NoAns; tie-break lower p50_ms)
    xs = df["F1"].fillna(0.0).tolist()
    ys = df["NoAns"].fillna(0.0).tolist()
    p50 = df["p50_ms"].fillna(1e9).tolist()

    frontier = []
    for i in range(len(xs)):
        dominated = False
        for j in range(len(xs)):
            if i == j: continue
            f1_better = xs[j] >= xs[i]
            na_better = ys[j] >= ys[i]
            strictly = (xs[j] > xs[i]) or (ys[j] > ys[i])
            if f1_better and na_better and strictly:
                dominated = True
                break
            if (xs[j] == xs[i]) and (ys[j] == ys[i]) and (p50[j] < p50[i]):
                dominated = True
                break
        if not dominated:
            frontier.append(i)

    # Scatter
    sizes = []
    for v in p50:
        try:
            v = max(1.0, float(v))
        except Exception:
            v = 1000.0
        sizes.append(2000.0 / (v ** 0.5))

    plt.figure()
    plt.scatter(xs, ys, s=sizes)
    # mark Pareto
    fx = [xs[i] for i in frontier]
    fy = [ys[i] for i in frontier]
    plt.scatter(fx, fy, marker="x", s=120)
    plt.xlabel("Overall F1")
    plt.ylabel("NoAns Accuracy")
    plt.title("Pareto: F1 vs NoAns (size ~ 1/√p50)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)

    print("\n=== Leaderboards ===")
    def show(title, key, asc=False, k=10):
        print("\n" + title)
        cols = ["file","F1","EM","AnsF1","NoAns","p50_ms","extractive","top_k","max_distance","null_threshold","rerank","rerank_lex_w","alpha","alpha_hits","support_min","support_window","span_max_distance"]
        print(df.sort_values(key, ascending=asc).head(k)[cols].to_string(index=False))
    show("Top 10 by Overall F1", "F1", asc=False)
    show("Top 10 by NoAns Accuracy", "NoAns", asc=False)
    show("Top 10 by Answerable F1", "AnsF1", asc=False)

    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_png}")
    return 0

def main():
    ap = argparse.ArgumentParser(description="Run a small grid of evaluations and build charts.")
    ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8000"))
    ap.add_argument("--key", default=os.environ.get("KEY", "my-dev-key-1"))
    ap.add_argument("--dataset", default=os.environ.get("DATASET", str(Path.cwd() / "data" / "squad" / "dev-v2.0.json")))
    ap.add_argument("--results", help="Output folder (default: results/<timestamp>)")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "500")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("TIMEOUT", "180")))
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = Path(args.results) if args.results else Path("results") / ts
    results.mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(parents=True, exist_ok=True)

    squad_eval_py = find_file(["scripts/squad_eval.py", "scripts\\squad_eval.py"])
    if not squad_eval_py:
        raise SystemExit("scripts/squad_eval.py not found. Please ensure it exists.")

    common = {
        "dataset": dataset,
        "url": args.url,
        "key": args.key,
        "limit": args.limit,
        "workers": args.workers,
        "timeout": args.timeout,
    }

    # ------- Grid definition (24 runs: 12 generative + 12 extractive) -------
    runs = []

    # helper to make numbers filename-safe
    def fmt(v: float) -> str:
        return f"{v:.2f}".replace(".", "p")

    # Generative grounded grid
    for top_k in [3, 5]:
        for nt in [0.20, 0.25, 0.30]:
            for rr in [0, 1]:
                cfg = dict(extractive=False, grounded_only=True, top_k=top_k, max_distance=0.60,
                           null_threshold=nt, temperature=0, rerank=bool(rr), rerank_lex_w=0.5)
                base = f"gen_k{top_k}_md{fmt(0.60)}_nt{fmt(nt)}_rr{rr}"
                name = base + ".json"
                runs.append((cfg, results / name))

    # Extractive grid
    for top_k in [3]:
        for nt in [0.55, 0.60, 0.65]:
            for rr in [0, 1]:
                for sd in [0.50, 0.60]:
                    cfg = dict(extractive=True, grounded_only=True, top_k=top_k, max_distance=0.60,
                               null_threshold=nt, temperature=0, rerank=bool(rr), rerank_lex_w=0.5,
                               alpha=0.50, alpha_hits=2, support_min=0.30, support_window=96,
                               span_max_distance=sd)
                    base = f"ex_k{top_k}_md{fmt(0.60)}_nt{fmt(nt)}_rr{rr}_sd{fmt(sd)}"
                    name = base + ".json"
                    runs.append((cfg, results / name))

    print(f"\nPlanned runs: {len(runs)}  (results -> {results})")
    ok = 0
    for i, (cfg, outp) in enumerate(runs, 1):
        outp.parent.mkdir(parents=True, exist_ok=True)
        success = run_eval(squad_eval_py, cfg, common, outp)
        ok += 1 if success else 0
        print(f"[Progress] {i}/{len(runs)} complete. OK so far: {ok}")

    if ok == 0:
        raise SystemExit("All runs failed; nothing to aggregate.")

    code = aggregate(results, Path("docs") / "combined_eval_table.csv", Path("docs") / "pareto_scatter.png")
    if code != 0:
        raise SystemExit(code)

if __name__ == "__main__":
    main()