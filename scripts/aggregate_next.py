# scripts/aggregate_next.py
#!/usr/bin/env python
"""
Aggregate SQuAD-style eval JSONs and plot a Pareto scatter.

Usage:
  python scripts/aggregate_next.py <INPUT_DIR> <OUTPUT_DIR>

Outputs:
  <OUTPUT_DIR>/combined_eval_table_next.csv
  <OUTPUT_DIR>/pareto_scatter_next.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Use a headless backend so saving figures works in containers/CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def get(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate eval JSONs and plot Pareto.")
    parser.add_argument("in_dir", type=Path, help="Directory containing *.json eval outputs")
    parser.add_argument("out_dir", type=Path, help="Directory to write CSV and PNG")
    args = parser.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fp in sorted(in_dir.glob("*.json")):
        try:
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            # Skip unreadable or malformed files
            continue

        o = obj.get("overall", {})
        a = obj.get("answerable", {})
        unans = obj.get("unanswerable", {})
        lat = obj.get("latency", {})
        s = obj.get("settings", {})

        rows.append(
            {
                "file": fp.name,
                "F1": get(o, "F1"),
                "EM": get(o, "EM"),
                "AnsF1": get(a, "F1"),
                "NoAns": get(unans, "NoAns_Accuracy"),
                "p50_ms": get(lat, "p50_ms"),
                "extractive": bool(get(s, "extractive", False)),
                "grounded_only": bool(get(s, "grounded_only", True)),
                "top_k": get(s, "top_k"),
                "max_distance": get(s, "max_distance"),
                "null_threshold": get(s, "null_threshold"),
                "rerank": bool(get(s, "rerank", False)),
                "rerank_lex_w": get(s, "rerank_lex_w"),
                "alpha": get(s, "alpha"),
                "alpha_hits": get(s, "alpha_hits"),
                "support_min": get(s, "support_min"),
                "support_window": get(s, "support_window"),
                "span_max_distance": get(s, "span_max_distance"),
            }
        )

    df = pd.DataFrame(rows)
    csvp = out_dir / "combined_eval_table_next.csv"
    df.to_csv(csvp, index=False)

    # Handle empty case gracefully
    if df.empty:
        # Still write an empty plot so downstream steps don't break
        plt.figure()
        plt.title("Pareto: F1 vs NoAns (size ~ 1/√p50)")
        plt.xlabel("Overall F1")
        plt.ylabel("NoAns Accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "pareto_scatter_next.png", dpi=160)
        plt.close()
        return 0

    # Coerce numeric columns & fill NaNs
    xs = pd.to_numeric(df["F1"], errors="coerce").fillna(0.0)
    ys = pd.to_numeric(df["NoAns"], errors="coerce").fillna(0.0)
    p50 = pd.to_numeric(df["p50_ms"], errors="coerce").fillna(1e9)

    # Compute Pareto front: keep points not dominated by any other
    front_idx = []
    X = xs.to_numpy()
    Y = ys.to_numpy()
    P = p50.to_numpy(dtype=float)

    n = len(df)
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j dominates i if >= on both and > on at least one
            if (X[j] >= X[i] and Y[j] >= Y[i]) and (X[j] > X[i] or Y[j] > Y[i]):
                dominated = True
                break
            # If equal on both metrics, prefer lower latency
            if X[j] == X[i] and Y[j] == Y[i] and P[j] < P[i]:
                dominated = True
                break
        if not dominated:
            front_idx.append(i)

    # Plot
    plt.figure()
    # larger marker for faster (smaller p50); add a floor to avoid div by zero
    sizes = 2000.0 / np.sqrt(np.maximum(1.0, P))
    plt.scatter(X, Y, s=sizes)
    if front_idx:
        plt.scatter(X[front_idx], Y[front_idx], marker="x", s=120)
    plt.xlabel("Overall F1")
    plt.ylabel("NoAns Accuracy")
    plt.title("Pareto: F1 vs NoAns (size ~ 1/√p50)")
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_scatter_next.png", dpi=160)
    plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())