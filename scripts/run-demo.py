"""Run the prepared demo queries end to end and keep their outputs as the fallback.

PLAN §8 asks for the demo to be run start to finish without intervention, and
for cached outputs to exist in case the machine is slow on the day. This does
both: every query goes through `indicrag ask` exactly as it would be typed, and
each response object is written to docs/demo/<n>-<slug>.json. Run it twice; the
second run reports whether any answer or citation changed.

    python scripts/run-demo.py                    # extractive answers
    python scripts/run-demo.py --gguf <model>     # generated answers, ~1 min each

The BM25 floor is on, at the dev-fitted 18.08 (paper §VI-H), so the
unanswerable query can be refused by the same gate the paper recommends.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo"
FLOOR = "18.08"

#: PLAN §8: English, Hindi, the brief's Hinglish example, a cross-lingual case,
#: a near-miss unanswerable, and a named entity that needs the alias table.
QUERIES = [
    ("english", "How much is the annual premium of Pradhan Mantri Suraksha Bima Yojana?"),
    ("hindi", "मनरेगा के तहत कितने दिन का रोजगार मिलता है?"),
    ("hinglish", "Scholarship ke liye minimum eligibility kya hai?"),
    ("cross-lingual", "सुकन्या समृद्धि खाता कितने वर्ष में परिपक्व होता है?"),
    ("near-miss", "What is the interest rate on a Kisan Credit Card loan in Nagaland?"),
    ("named-entity", "PMJDY ke tahat kitne bank khate khole gaye?"),
]


def ask(query: str, gguf: str) -> dict:
    cmd = [sys.executable, "-m", "indicrag.cli", "ask", query, "--bm25-floor", FLOOR]
    if gguf:
        cmd += ["--gguf", gguf]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip().splitlines()[-1] if proc.stderr else "ask failed")
    return json.loads(proc.stdout[proc.stdout.index("{"):])


def summary(result: dict) -> tuple:
    return (
        result["answerability"],
        result["answer"],
        tuple(c["passage_id"] for c in result["citations"]),
    )


def main() -> int:
    gguf = sys.argv[sys.argv.index("--gguf") + 1] if "--gguf" in sys.argv else ""
    OUT.mkdir(parents=True, exist_ok=True)
    changed = 0
    for n, (slug, query) in enumerate(QUERIES, start=1):
        path = OUT / f"{n}-{slug}.json"
        before = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        start = time.time()
        result = ask(query, gguf)
        result.pop("latency_ms", None)  # varies run to run; not part of the answer
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        same = before is not None and summary(before) == summary(result)
        changed += before is not None and not same
        top = result["citations"][0]["passage_id"] if result["citations"] else "-"
        answer = re.sub(r"\s+", " ", result["answer"])[:70]
        print(f"{n}. {slug:<14} {time.time() - start:5.1f}s  {result['query_type']:<10} "
              f"{result['answerability']:<12} top={top}"
              + ("" if before is None else ("  (unchanged)" if same else "  (CHANGED)")))
        print(f"   {answer}")
    print(f"\n{len(QUERIES)} queries -> {OUT.relative_to(ROOT)}"
          + (f"; {changed} changed since the last run" if changed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
