"""Regenerate every committed report, or check that each one still regenerates.

PRD NFR-6 says every reported table is reproducible by one command, and names
`indicrag eval all --report evals/` as that command. It is not: `eval all`
writes four files, and only two of them are among the reports committed under
`evals/`. The rest came from `check-*.py` scripts, from single CLI commands run
by hand with particular flags, or from scripts no longer in the repository --
and which was which was recorded nowhere. A report that cannot be traced to a
command cannot be checked, and the paper quoted numbers from sections that no
committed report contained.

So this is the table: every committed report, the command that produces it, and
what that command needs. Two modes:

    python scripts/regenerate-reports.py --check          # regenerate to a scratch dir, diff
    python scripts/regenerate-reports.py --write          # regenerate into evals/

`--check` ignores the `Generated:` timestamp and path separators in `Source:`,
and nothing else. A report whose command needs something absent -- a GGUF
generator, an encoder index -- is reported as skipped with the reason, never as
passing. Two reports have no producer at all and are listed as legacy.

It runs against whatever INDICRAG_DATA_DIR holds. Every committed number is
measured on the frozen corpus, so build that first:

    indicrag corpus fetch                 # check the sha256s match the manifest
    indicrag corpus segment               # --version 1, the frozen 694 passages
    indicrag corpus repair-spans --apply  # spans only; IDs and text unchanged
    indicrag index lexical
    indicrag index build --all            # MuRIL: add --pooling cls for its second row
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
PY = sys.executable
CRLF = b"\r\n"


@dataclass(frozen=True)
class Report:
    name: str
    #: argv; "{out}" is the output path, "{tmp}" a scratch directory, "{gguf}"
    #: the generator. None for a report with no producer.
    argv: tuple[str, ...] | None
    #: The command prints the report to stdout rather than taking --report.
    stdout: bool = False
    #: Index file stems that must exist under emb/, or "gguf".
    needs: tuple[str, ...] = ()
    note: str = ""


def _cli(*args: str) -> tuple[str, ...]:
    return (PY, "-m", "indicrag.cli", *args)


def _script(name: str) -> tuple[str, ...]:
    return (PY, str(ROOT / "scripts" / name))


_ERRORS = """
from pathlib import Path
from indicrag.config import get_settings
from indicrag.evaluation.all_reports import provenance
from indicrag.evaluation.errors import ErrorCase, format_errors
from indicrag.models import Passage, QAItem, read_jsonl
items = list(read_jsonl(Path("evals/gold.jsonl"), QAItem))
passages = list(read_jsonl(get_settings().passages_path, Passage))
cases = [ErrorCase.from_dict(d) for d in read_jsonl(Path("evals/errors.jsonl"))]
print("\\n".join(provenance(items, "evals/gold.jsonl") + format_errors(cases, passages)))
"""

#: Index file stems under emb/, as `config.slugify` writes them.
E5 = "intfloat__multilingual-e5-base"
MINILM = "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2"
LABSE = "sentence-transformers__LaBSE"
MURIL = "google__muril-base-cased"
ALL_DENSE = (E5, MINILM, LABSE, MURIL)

REPORTS: tuple[Report, ...] = (
    Report("report-corpus-audit.txt", _cli("corpus", "audit", "--report", "{out}"),
           note="needs spans repaired: `corpus repair-spans --apply`"),
    Report("report-dataset-coverage.txt", _cli("dataset", "stats", "--report", "{out}")),
    Report("report-second-pass.txt",
           _cli("dataset", "second-pass", "--compare", "evals/second-pass.jsonl",
                "--report", "{out}")),
    Report("report-retrieval-goldset.txt", _cli("eval", "retrieval", "--report", "{out}"),
           needs=ALL_DENSE),
    # P5: the headline retrieval table on the sealed test split, scored once.
    # report-retrieval-goldset.txt is the same harness over dev and test together.
    Report("report-retrieval-test.txt",
           _cli("eval", "retrieval", "--split", "test", "--report", "{out}"),
           needs=ALL_DENSE),
    Report("report-answerability.txt", _cli("eval", "answerability", "--report", "{out}"),
           needs=(E5,)),
    Report("report-answerability-bm25.txt",
           _cli("eval", "answerability", "--method", "bm25", "--report", "{out}")),
    Report("report-answerability-signals.txt",
           _cli("eval", "answerability", "--gguf", "{gguf}",
                "--sample", "120", "--enrich", "--report", "{out}"),
           needs=("gguf",),
           note="needs the Qwen2.5-3B GGUF and llama-cpp-python"),
    # Formats the committed evals/errors.jsonl, as `eval all`'s error stage does:
    # those are the 20 probe-era cases §VIII's account rests on. `eval errors`
    # re-selects cases on the current gold set, which is a different experiment
    # and would silently replace the evidence the paper cites.
    Report("report-errors.txt", (PY, "-c", _ERRORS), stdout=True),
    # P5: Module 4 on a 72-item stratified sample of the sealed test split.
    Report("report-qa.txt",
           _cli("eval", "qa", "--split", "test", "--sample", "72", "--nli",
                "--gguf", "{gguf}", "--report", "{out}"),
           needs=("gguf",),
           note="needs the Qwen2.5-3B GGUF and llama-cpp-python"),
    # Module 6 re-selected on the sealed test split, over the paper's system.
    # Its own files: report-errors.txt keeps the probe-era cases §VII-C cites.
    Report("report-errors-test.txt",
           _cli("eval", "errors", "--split", "test", "--system", "Hybrid RRF script-aware",
                "--out", "{tmp}/errors-test.jsonl", "--report", "{out}"),
           needs=ALL_DENSE),
    Report("report-amount-matching.txt", _script("check-amount-matching.py"), stdout=True),
    Report("report-leakage.txt", _script("check-leakage.py"), stdout=True, needs=(E5,)),
    Report("report-fusion-goldset.txt", _script("check-fusion-goldset.py"), stdout=True,
           needs=(E5,)),
    # P5: the paper's central comparison on the sealed test split, scored once.
    Report("report-fusion-test.txt", (*_script("check-fusion-goldset.py"), "--split", "test"),
           stdout=True, needs=(E5,)),
    # A defect found in the test-split error cases, decided on dev (not adopted).
    Report("report-observed-votes.txt", _script("check-observed-votes.py"), stdout=True,
           needs=(E5,)),
    Report("report-fusion-normalisation.txt", _script("check-fusion-normalisation.py"),
           stdout=True, needs=(E5,)),
    Report("report-fusion-encoders.txt", _script("check-fusion-encoders.py"), stdout=True,
           needs=(E5, MINILM, LABSE)),
    Report("report-encoder-comparisons.txt", _script("check-labse-claim.py"), stdout=True,
           needs=(E5, LABSE)),
    Report("report-retrieval-probes.txt", None,
           note="assembled on 2026-09-20 by a script not in the repository; its alpha "
           "sweep was re-run by hand on 2026-09-23. `eval retrieval --gold <absent>` "
           "measures the same probes in the current format"),
    Report("report-review.txt", None,
           note="half judgement: `dataset review` re-runs the rule checks, but the reading "
           "pass is a reviewer's notes, not a computation; they are kept per item in "
           "evals/review-assist.jsonl"),
    Report("report-script-aware-fusion.txt", None,
           note="from a one-off probe script not in the repository; the same comparison "
           "on the gold set is report-fusion-goldset.txt"),
)


def _normalise(text: str) -> list[str]:
    """Drop the timestamp; unify path separators in the Source line."""
    out = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("Generated:"):
            continue
        if line.startswith("Source:"):
            line = line.replace("\\", "/")
        out.append(line)
    return out


def _missing(report: Report, emb_dir: Path, gguf: str) -> str | None:
    for need in report.needs:
        if need == "gguf":
            if not gguf or not Path(gguf).exists():
                return "no GGUF generator (pass --gguf)"
        elif not any(emb_dir.glob(f"{need}*.npy")):
            return f"no index for {need} under {emb_dir}"
    return None


def _run(report: Report, out: Path, tmp: Path, gguf: str) -> str | None:
    argv = [
        a.replace("{out}", str(out)).replace("{tmp}", str(tmp)).replace("{gguf}", gguf)
        for a in report.argv or ()
    ]
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
    }
    proc = subprocess.run(
        argv, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return f"exit {proc.returncode}: " + " | ".join(tail)
    if report.stdout:
        # stdout only: the committed reports that carried a Hub warning and a
        # progress bar in their first lines got them from a redirected stderr.
        out.write_text(proc.stdout, encoding="utf-8", newline="\n")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="regenerate to a scratch dir and diff")
    mode.add_argument("--write", action="store_true", help="regenerate into evals/")
    ap.add_argument("--only", nargs="*", default=None, help="report names to include")
    ap.add_argument("--gguf", default=os.environ.get("INDICRAG_LLM_GGUF_PATH", ""))
    ap.add_argument("--diff-lines", type=int, default=12)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from indicrag.config import get_settings

    emb_dir = get_settings().emb_dir
    reports = [r for r in REPORTS if args.only is None or r.name in args.only]
    failed = 0
    with tempfile.TemporaryDirectory(prefix="indicrag-regen-") as scratch:
        tmp = Path(scratch)
        for r in reports:
            if r.argv is None:
                print(f"  LEGACY    {r.name:<36} {r.note}")
                continue
            why = _missing(r, emb_dir, args.gguf)
            if why:
                print(f"  SKIPPED   {r.name:<36} {why}")
                continue
            out = (EVALS if args.write else tmp) / r.name
            crlf = out.exists() and CRLF in out.read_bytes()
            err = _run(r, out, tmp, args.gguf)
            if err:
                failed += 1
                print(f"  FAILED    {r.name:<36} {err}")
                continue
            if args.write:
                if crlf:
                    # Keep the committed file's line endings, or every line
                    # shows as changed and the real difference is unreadable.
                    body = out.read_bytes().replace(CRLF, b"\n")
                    out.write_bytes(body.replace(b"\n", CRLF))
                print(f"  WRITTEN   {r.name}")
                continue
            want = _normalise((EVALS / r.name).read_text(encoding="utf-8"))
            got = _normalise(out.read_text(encoding="utf-8"))
            if want == got:
                print(f"  OK        {r.name}")
                continue
            failed += 1
            diff = list(
                difflib.unified_diff(want, got, "committed", "regenerated", n=0, lineterm="")
            )
            changed = sum(1 for d in diff[2:] if d[:1] in "+-")
            print(f"  DIFFERS   {r.name}  ({changed} lines)")
            for line in diff[2 : 2 + args.diff_lines]:
                print(f"              {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
