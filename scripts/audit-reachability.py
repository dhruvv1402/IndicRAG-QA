"""Find public code that nothing calls.

This project's characteristic failure is not a crash. It is a function that is
written, tested in isolation, and then never wired to anything -- after which
the report it was meant to fill prints a dash, or a zero, or silently falls back,
and nobody notices because nothing errors.

Six of those were found by hand over one session: `SelfReportSignal`,
`NLIScorer`, `LlamaCppProvider.answer`, `cited_found`, `alpha_sweep` and
`load_probes`. Two of them fed numbers the paper quotes. This script is the
audit that found the last two, kept so it can be re-run instead of rediscovered.

    python scripts/audit-reachability.py

Exits non-zero when something unexpected is unreachable, so it can gate a
commit. Entries in ALLOWED are the known-good cases, each with the reason it is
not a defect; anything else is reported.
"""

from __future__ import annotations

import ast
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "indicrag"
TESTS = ROOT / "tests"

#: Reachable despite having no textual caller, with the reason why.
ALLOWED: dict[str, str] = {
    # --- protocols and settings -------------------------------------------
    "Provider": "Protocol, used as a structural type",
    "Settings": "pydantic-settings model, built by get_settings()",
    # --- dataclasses constructed and returned within their own module -----
    "Agreement": "returned by compare()",
    "Check": "returned by the validate helpers",
    "ConfusionMatrix": "returned by AnswerabilityReport.confusion()",
    "Coverage": "returned by coverage()",
    "FetchError": "raised, not called",
    "FetchResult": "returned by fetch_corpus()",
    "Generated": "returned by every Provider.answer()",
    "GroundingOutcome": "returned by score_generation()",
    "IntegrityReport": "returned by validate()",
    "Module4Result": "returned by run_module4()",
    "PairedResult": "returned by paired_test()",
    "Progress": "returned by the verify loop",
    "Section": "returned by split_sections()",
    # --- called only from within their own module -------------------------
    "best_over_golds": "used by QAOutcome scoring",
    "build_hinglish_prompt": "used by the dataset generator",
    "build_prompt_for": "used by run_arm()",
    "content_hash": "used by the models' id derivation",
    "count_tokens": "used by segmentation",
    "find_final_viramas": "used by validate()",
    "find_misplaced_matras": "used by validate()",
    "has_devanagari": "used by to_roman()",
    "is_code_mixed": "used by classify()",
    "is_devanagari_token": "used by the tokenizer",
    "normalize_punctuation": "used by normalize()",
    "normalize_whitespace": "used by normalize()",
    "parse_prompt": "used by extractive_complete()",
    "parse_translation": "used by the dataset generator",
    "passages_fingerprint": "used by DenseIndex save/load",
    "render_item": "used by the verify loop",
    "resolve_pairs": "used by fetch_corpus()",
    "segment_all": "used by segment_document()",
    "slugify": "used by config path derivation",
    "split_sections": "used by segmentation",
    "to_roman": "used by the transliteration bridge",
    "token_overlap": "used by probe construction",
    # --- known dead, deliberately ------------------------------------------
    "format_misses": "no caller and no artefact depends on it; left rather than wired for its own sake",
}


def public_definitions() -> dict[str, tuple[str, str, bool]]:
    """name -> (file, kind, is_cli_command)."""
    found: dict[str, tuple[str, str, bool]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith("_"):
                continue
            # A Typer command is reached through its decorator, never by name.
            decorated = any(
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr == "command"
                for d in getattr(node, "decorator_list", [])
            )
            rel = str(path.relative_to(ROOT))
            found.setdefault(node.name, (rel, type(node).__name__, decorated))
    return found


def external_uses(names: set[str]) -> collections.Counter:
    """How often each name is referenced from a file other than its own."""
    uses: collections.Counter = collections.Counter()
    for path in list(SRC.rglob("*.py")) + list(TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.alias):
                name = node.name.split(".")[-1]
            if name in names and DEFS[name][0] != owner:
                uses[name] += 1
    return uses


def main() -> int:
    global DEFS
    DEFS = public_definitions()
    uses = external_uses(set(DEFS))

    unreachable = []
    for name, (path, kind, is_command) in sorted(DEFS.items()):
        if uses[name] or is_command or name in ALLOWED:
            continue
        unreachable.append((name, path, kind))

    stale = sorted(n for n in ALLOWED if n not in DEFS)

    print(f"{len(DEFS)} public top-level definitions in {SRC.relative_to(ROOT)}")
    print(f"{sum(1 for v in DEFS.values() if v[2])} are CLI commands (reached by decorator)")
    print(f"{len(ALLOWED) - len(stale)} are allow-listed as reachable from within their module")
    print()

    if stale:
        print("ALLOWED names that no longer exist -- remove them:")
        for name in stale:
            print(f"  {name}")
        print()

    if not unreachable:
        print("No unexpected unreachable definitions.")
        return 1 if stale else 0

    print(f"{len(unreachable)} definition(s) nothing outside their own module references:")
    print()
    for name, path, kind in unreachable:
        print(f"  {kind:<11} {name:<32} {path}")
    print()
    print("Each is either dead code, or working code that was never wired to the")
    print("thing it was written for. The second kind does not fail loudly: the")
    print("report it should fill prints a dash, a zero, or a fallback instead.")
    print("Wire it, delete it, or add it to ALLOWED with the reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
