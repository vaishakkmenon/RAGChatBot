# scripts/squad_to_md.py
from __future__ import annotations
import argparse, json, hashlib, os, re, unicodedata
from pathlib import Path

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return re.sub(r"-{2,}", "-", text)[:64] or "ctx"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to dev-v2.0.json")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder for .md files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = 0
    for art in data.get("data", []):
        title = art.get("title", "untitled")
        title_slug = slugify(title)
        for para in art.get("paragraphs", []):
            ctx = (para.get("context") or "").strip()
            if not ctx:
                continue
            h = hashlib.sha256(ctx.encode("utf-8")).hexdigest()[:12]
            fn = f"{title_slug}-{h}.md"
            p = out_dir / fn
            # Simple markdown body: title header + context
            p.write_text(f"# {title}\n{ctx}\n", encoding="utf-8")
            n += 1
    print("wrote", n, "markdown files to", str(out_dir))

if __name__ == "__main__":
    main()