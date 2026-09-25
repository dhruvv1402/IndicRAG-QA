"""Assemble a Hugging Face Docker Space for the web interface.

The Space needs a small, exact subset of this repository: the package, the web
page, the paper files, the rehearsed demo outputs, and the frozen corpus with
its lexical index and the primary (e5) embeddings -- about 5 MB of data. This
copies exactly that into one folder with deploy/hf-space/'s Dockerfile and
README, then loads the copied data the way the server will, so a bundle whose
index does not match its passages fails here rather than on the Space.

    python scripts/build-space.py                  # -> build/hf-space/
    python scripts/build-space.py --out <dir>

It reads the corpus from INDICRAG_DATA_DIR (run scripts/dev-env first). To
publish, push the folder to a Space you created (see deploy/hf-space/README.md
and docs/DEMO.md): the folder is the Space repository's whole content.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PAPER_FILES = ("IndicRAG-QA.pdf", "IndicRAG-QA-IEEE.pdf", "IndicRAG-QA-slides.pptx")
E5 = "intfloat__multilingual-e5-base"


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "hf-space")
    args = ap.parse_args()

    from indicrag.config import get_settings

    cfg = get_settings()
    data = cfg.passages_path.parent
    needed = [
        cfg.passages_path, cfg.manifest_path,
        cfg.lex_dir / "lexical.pkl", cfg.lex_dir / "lexical.meta.json",
        cfg.emb_dir / f"{E5}.npy", cfg.emb_dir / f"{E5}.meta.json",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print("missing corpus files (set INDICRAG_DATA_DIR, e.g. run scripts/dev-env):")
        for m in missing:
            print(f"  {m}")
        return 1

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in ("Dockerfile", "README.md"):
        _copy(ROOT / "deploy" / "hf-space" / name, out / name)
    _copy(ROOT / "pyproject.toml", out / "pyproject.toml")
    _copy(ROOT / "LICENSE", out / "LICENSE")
    shutil.copytree(ROOT / "src" / "indicrag", out / "src" / "indicrag",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "web", out / "web", ignore=shutil.ignore_patterns("_*"))
    for name in PAPER_FILES:
        _copy(ROOT / "paper" / name, out / "paper" / name)
    for p in sorted((ROOT / "docs" / "demo").glob("*.json")):
        _copy(p, out / "docs" / "demo" / p.name)
    for p in needed:
        _copy(p, out / "data" / p.relative_to(data))
    (out / ".dockerignore").write_text("**/__pycache__\n*.pyc\n", encoding="utf-8")

    # Load the bundle's own data exactly as the server will.
    from indicrag.index.dense import DenseIndex
    from indicrag.index.encoders import get
    from indicrag.index.lexical import LexicalIndex
    from indicrag.models import Passage, read_jsonl

    passages = list(read_jsonl(out / "data" / cfg.passages_path.name, Passage))
    LexicalIndex.load(out / "data" / cfg.lex_dir.name)
    DenseIndex.load(get(cfg.encoder_primary), out / "data" / cfg.emb_dir.name, passages)

    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    print(f"wrote {out}  ({len(files)} files, {size / 1e6:.1f} MB; {len(passages)} passages, "
          f"indices verified against them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
