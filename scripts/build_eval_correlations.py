#!/usr/bin/env python3
"""
Build correlation plots/tables from existing evaluation CSV.

- Loads docs/combined_eval_table_next.csv if present, else docs/combined_eval_table.csv.
- Cleans dtypes, tags mode (generative/extractive), and computes:
  * corr_summary_next.csv (Pearson & Spearman against key knobs)
  * A set of single-axis matplotlib PNGs (no seaborn, no subplots, no explicit colors)
  * Optional Pareto re-plot (F1 vs NoAns; tie-break lower p50_ms)
  * README-friendly Top-5 per mode tables (saved to docs/top5_tables_next.md)

Usage (from repo root, inside container):
  python /workspace/scripts/build_eval_correlations.py
Options:
  --csv /workspace/docs/combined_eval_table_next.csv
  --out-dir /workspace/docs
  --pareto
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- helpers -----------------------------
def _series_or_default(df: pd.DataFrame, col: str, default_value: float | bool | None = np.nan) -> pd.Series:
    """Return a Series aligned to df.index; use default_value if the column is missing."""
    if col in df.columns:
        s = df[col]
        # Ensure alignment
        if not isinstance(s, pd.Series) or not s.index.equals(df.index):
            s = pd.Series(s.values, index=df.index)
    else:
        s = pd.Series([default_value] * len(df), index=df.index)
    return s


def _to_bool(x: object) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n"):
        return False
    return None


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    # numeric knobs/metrics
    num_cols = [
        "F1", "EM", "AnsF1", "NoAns", "p50_ms", "top_k", "max_distance", "null_threshold",
        "rerank_lex_w", "alpha", "alpha_hits", "support_min", "support_window", "span_max_distance"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # booleans → normalize
    for c in ("extractive", "grounded_only", "rerank"):
        if c in df.columns:
            df[c] = df[c].map(_to_bool)

    # derived flags (always produce aligned Series)
    extractive_s = _series_or_default(df, "extractive", False).map(lambda v: bool(v) if v is not None else False)
    rerank_s = _series_or_default(df, "rerank", False).map(lambda v: bool(v) if v is not None else False)

    df["mode"] = np.where(extractive_s, "extractive", "generative")
    df["rerank_flag"] = rerank_s.astype(int)
    df["extractive_flag"] = extractive_s.astype(int)
    return df


def _scatter_save(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    overlay_by_mode: bool = False,
    filter_expr: Optional[str] = None,
    trend: bool = True,
) -> Optional[str]:
    d = df.copy()
    if filter_expr:
        try:
            d = d.query(filter_expr)
        except Exception:
            # if bad filter, skip plot gracefully
            return None

    keep = [c for c in (x, y, "mode") if c in d.columns]
    d = d[keep].dropna()
    if d.empty:
        return None

    plt.figure()
    if overlay_by_mode and "mode" in d.columns:
        for mode, marker in (("generative", "o"), ("extractive", "x")):
            dd = d[d["mode"] == mode]
            if not dd.empty:
                plt.scatter(dd[x], dd[y], marker=marker, label=mode)
        plt.legend()
    else:
        plt.scatter(d[x], d[y])

    if trend and len(d) >= 2:
        try:
            coeffs = np.polyfit(d[x].astype(float), d[y].astype(float), 1)
            xs = np.linspace(float(d[x].min()), float(d[x].max()), 100)
            ys = coeffs[0] * xs + coeffs[1]
            plt.plot(xs, ys)
        except Exception:
            pass

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return str(out_path)


def _corr_summary(df: pd.DataFrame, out_csv: Path) -> Path:
    metrics = ["F1", "NoAns", "AnsF1", "p50_ms"]
    knobs = [
        "top_k", "max_distance", "null_threshold", "rerank_flag", "rerank_lex_w",
        "alpha", "alpha_hits", "support_min", "support_window", "span_max_distance",
    ]
    rows: list[dict] = []
    for m in metrics:
        if m not in df.columns:
            continue
        for k in knobs:
            if k not in df.columns:
                continue
            dsub = df[[m, k]].dropna()
            if len(dsub) < 3:
                continue
            pear = dsub[m].corr(dsub[k], method="pearson")
            spear = dsub[m].corr(dsub[k], method="spearman")
            rows.append({
                "metric": m,
                "var": k,
                "pearson_r": round(float(pear), 4) if pd.notna(pear) else None,
                "spearman_rho": round(float(spear), 4) if pd.notna(spear) else None,
                "n": int(len(dsub)),
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["metric", "var"]).to_csv(out_csv, index=False)
    return out_csv


def _top5_tables(df: pd.DataFrame, md_path: Path) -> Path:
    def top5(is_extractive: bool) -> pd.DataFrame:
        f = df[df["extractive_flag"] == (1 if is_extractive else 0)].copy()
        if f.empty:
            return f
        return f.sort_values(
            by=["F1", "NoAns", "AnsF1", "p50_ms"],
            ascending=[False, False, False, True],
        ).head(5)

    cols = [
        "F1", "EM", "NoAns", "AnsF1", "p50_ms", "top_k", "max_distance", "null_threshold",
        "rerank", "rerank_lex_w", "alpha", "alpha_hits", "support_min", "support_window", "span_max_distance", "file",
    ]
    gen = top5(False)[[c for c in cols if c in df.columns]]
    ex = top5(True)[[c for c in cols if c in df.columns]]

    def to_md(title: str, d: pd.DataFrame) -> str:
        if d.empty:
            return f"### {title}\n_No rows_\n"
        d2 = d.copy()
        for c in ["F1", "EM", "NoAns", "AnsF1", "max_distance", "null_threshold", "rerank_lex_w", "alpha", "support_min", "span_max_distance"]:
            if c in d2:
                d2[c] = pd.to_numeric(d2[c], errors="coerce").round(3)
        lines = [f"### {title}",
                 "| " + " | ".join(d2.columns.astype(str)) + " |",
                 "|" + "|".join(["---"] * len(d2.columns)) + "|"]
        for _, r in d2.iterrows():
            vals = ["" if pd.isna(v) else str(v) for v in r]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    md = to_md("Top 5 Generative", gen) + "\n\n" + to_md("Top 5 Extractive", ex)
    md_path.write_text(md, encoding="utf-8")
    return md_path


def _pareto_plot(df: pd.DataFrame, out_png: Path) -> Path:
    xs = df["F1"].fillna(0.0).to_numpy()
    ys = df["NoAns"].fillna(0.0).to_numpy()
    p50 = df["p50_ms"].fillna(1e9).to_numpy()
    frontier: list[int] = []
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            if (xs[j] >= xs[i] and ys[j] >= ys[i]) and (xs[j] > xs[i] or ys[j] > ys[i]):
                dominated = True
                break
            if (xs[j] == xs[i]) and (ys[j] == ys[i]) and (p50[j] < p50[i]):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    sizes = 2000.0 / (np.sqrt(np.clip(p50, 1.0, None)))
    plt.figure()
    plt.scatter(xs, ys, s=sizes)
    plt.scatter(xs[frontier], ys[frontier], marker="x", s=120)
    plt.xlabel("Overall F1")
    plt.ylabel("NoAns Accuracy")
    plt.title("Pareto: F1 vs NoAns (size ~ 1/√p50)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png


# ----------------------------- main -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to combined_eval_table(_next).csv")
    ap.add_argument("--out-dir", default="docs", help="Folder for PNG/CSV/MD outputs")
    ap.add_argument("--pareto", action="store_true", help="Also (re)build Pareto plot")
    args = ap.parse_args()

    # Resolve CSV
    if args.csv:
        src = Path(args.csv)
    else:
        cands = [Path("docs/combined_eval_table_next.csv"), Path("docs/combined_eval_table.csv")]
        src = next((p for p in cands if p.exists()), None)
        if src is None:
            raise SystemExit(
                "No combined_eval_table CSV found. Expected docs/combined_eval_table_next.csv or docs/combined_eval_table.csv"
            )

    df = pd.read_csv(src)
    df = _coerce(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Correlation summary
    corr_csv = out_dir / "corr_summary_next.csv"
    _corr_summary(df, corr_csv)

    # 2) Plots
    artifacts: list[str] = []
    S = lambda fname: out_dir / fname

    # helper to satisfy the type checker
    def _add_if(p: Optional[str]) -> None:
        if p is not None:
            artifacts.append(p)

    _add_if(_scatter_save(df, "null_threshold", "F1", S("corr_F1_vs_null_threshold.png"),
                        "F1 vs Null Threshold (all)", "Null Threshold", "F1",
                        overlay_by_mode=True, trend=True))

    _add_if(_scatter_save(df, "null_threshold", "NoAns", S("corr_NoAns_vs_null_threshold.png"),
                        "NoAns vs Null Threshold (all)", "Null Threshold", "NoAns",
                        overlay_by_mode=True, trend=True))

    _add_if(_scatter_save(df, "null_threshold", "AnsF1", S("corr_AnsF1_vs_null_threshold.png"),
                        "Answerable F1 vs Null Threshold (all)", "Null Threshold", "Answerable F1",
                        overlay_by_mode=True, trend=True))

    _add_if(_scatter_save(df, "max_distance", "F1", S("corr_F1_vs_max_distance.png"),
                        "F1 vs Max Distance (all)", "Max Distance", "F1",
                        overlay_by_mode=True, trend=True))

    _add_if(_scatter_save(df, "top_k", "F1", S("corr_F1_vs_top_k.png"),
                        "F1 vs top_k (all)", "top_k", "F1",
                        overlay_by_mode=True, trend=True))

    _add_if(_scatter_save(df, "top_k", "p50_ms", S("corr_p50_vs_top_k.png"),
                        "Latency p50 vs top_k (all)", "top_k", "p50_ms",
                        overlay_by_mode=True, trend=True))

    _add_if(_scatter_save(df, "rerank_lex_w", "F1", S("corr_F1_vs_rerank_lex_w.png"),
                        "F1 vs rerank_lex_w (rerank=1)", "rerank_lex_w", "F1",
                        overlay_by_mode=True, filter_expr="rerank_flag == 1", trend=True))

    # extractive-only knobs
    _add_if(_scatter_save(df, "alpha", "F1", S("corr_F1_vs_alpha.png"),
                        "F1 vs alpha (extractive)", "alpha", "F1",
                        overlay_by_mode=False, filter_expr="extractive_flag == 1", trend=True))

    _add_if(_scatter_save(df, "support_min", "F1", S("corr_F1_vs_support_min.png"),
                        "F1 vs support_min (extractive)", "support_min", "F1",
                        overlay_by_mode=False, filter_expr="extractive_flag == 1", trend=True))

    _add_if(_scatter_save(df, "support_window", "AnsF1", S("corr_AnsF1_vs_support_window.png"),
                        "Answerable F1 vs support_window (extractive)", "support_window", "AnsF1",
                        overlay_by_mode=False, filter_expr="extractive_flag == 1", trend=True))

    _add_if(_scatter_save(df, "span_max_distance", "F1", S("corr_F1_vs_span_max_distance.png"),
                        "F1 vs span_max_distance (extractive)", "span_max_distance", "F1",
                        overlay_by_mode=False, filter_expr="extractive_flag == 1", trend=True))

    # 3) Heatmap (drop columns that are all-NaN or constant)
    heat_cols = [
        "F1", "EM", "AnsF1", "NoAns", "p50_ms", "top_k", "max_distance", "null_threshold",
        "rerank_flag", "rerank_lex_w", "alpha", "alpha_hits", "support_min", "support_window", "span_max_distance",
    ]
    heat_avail = [c for c in heat_cols if c in df.columns]
    heat_df = df[heat_avail].copy()
    # remove constant columns (nunique <= 1) to avoid NaN-only rows/cols in corr
    const_cols = [c for c in heat_df.columns if heat_df[c].nunique(dropna=True) <= 1]
    heat_df.drop(columns=const_cols, inplace=True, errors="ignore")

    cmat = heat_df.corr(method="pearson")
    plt.figure()
    plt.imshow(cmat, aspect="auto")
    # Pylance-friendly: lists of str for labels
    xt_labels: list[str] = [str(c) for c in cmat.columns]
    yt_labels: list[str] = [str(i) for i in cmat.index]
    plt.xticks(ticks=range(len(xt_labels)), labels=xt_labels, rotation=90)
    plt.yticks(ticks=range(len(yt_labels)), labels=yt_labels)
    plt.colorbar()
    plt.title("Correlation Matrix (Pearson)")
    plt.tight_layout()
    heat_path = out_dir / "corr_matrix_pearson.png"
    plt.savefig(heat_path, dpi=160)
    plt.close()
    artifacts.append(str(heat_path))

    # 4) README tables
    md_path = out_dir / "top5_tables_next.md"
    _top5_tables(df, md_path)

    # 5) Optional Pareto
    if args.pareto:
        _pareto_plot(df, out_dir / "pareto_scatter_next.png")

    print("Artifacts written:")
    print(" -", corr_csv.resolve())
    for a in artifacts:
        if a:
            print(" -", Path(a).resolve())
    print(" -", md_path.resolve())
    if args.pareto:
        print(" -", (out_dir / "pareto_scatter_next.png").resolve())


if __name__ == "__main__":
    main()