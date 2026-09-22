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
    "Span": "returned by corpus.align.locate()",
    # --- dataclasses constructed and returned within their own module -----
    "Agreement": "returned by compare()",
    "AlphaSweep": "returned by alpha_sweep()",
    "Check": "returned by the validate helpers",
    "Citation": "one entry in Reanchored.citations, built by reanchor()",
    "ConfusionMatrix": "returned by AnswerabilityReport.confusion()",
    "CorpusAudit": "returned by corpus.integrity.audit()",
    "Coverage": "returned by coverage()",
    "FetchError": "raised, not called",
    "Finding": "one entry in CorpusAudit.findings, built by audit()",
    "FetchResult": "returned by fetch_corpus()",
    "Generated": "returned by every Provider.answer()",
    "GroundingOutcome": "returned by score_generation()",
    "IntegrityReport": "returned by validate()",
    "Module4Result": "returned by run_module4()",
    "PairedResult": "returned by paired_test()",
    "ReanchorResult": "returned by dataset.reanchor.reanchor()",
    "Reanchored": "one entry in ReanchorResult.items, built by reanchor()",
    "RepairResult": "returned by corpus.align.repair_spans()",
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


def external_uses(names: set[str], roots, *, same_file: bool = False) -> collections.Counter:
    """How often each name is referenced.

    `same_file` includes references from the defining module. The unreachable
    check excludes them, because a helper called only by its own module is
    fine and is allow-listed. The test-only check must include them, or every
    such helper looks test-only the moment a test imports it.

    A `def` is a FunctionDef node rather than a Name, so a definition never
    counts as a reference to itself.
    """
    uses: collections.Counter = collections.Counter()
    for root in roots:
        for path in root.rglob("*.py"):
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
                if name in names and (same_file or DEFS[name][0] != owner):
                    uses[name] += 1
    return uses


def unimplemented_methods() -> list[tuple[str, str, int]]:
    """Public methods whose body is `raise NotImplementedError`.

    The module-level scan cannot see these -- `LlamaCppProvider.answer` was one,
    deferring to a phase that had already shipped, which meant the demo could
    only ever run the extractive fallback. A Protocol's `...` stubs are the
    legitimate form and are not flagged; a concrete class that raises is either
    abstract by convention or a gap nobody closed.
    """
    out: list[tuple[str, str, int]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            # Protocols and ABCs declare stubs on purpose.
            bases = {
                b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                for b in cls.bases
            }
            if bases & {"Protocol", "ABC"}:
                continue
            for node in cls.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                body = [n for n in node.body if not isinstance(n, ast.Expr)]
                if (
                    len(body) == 1
                    and isinstance(body[0], ast.Raise)
                    and "NotImplementedError" in ast.dump(body[0])
                ):
                    out.append((f"{cls.name}.{node.name}", str(path.relative_to(ROOT)), node.lineno))
    return out


def main() -> int:
    global DEFS
    DEFS = public_definitions()
    names = set(DEFS)
    src_uses = external_uses(names, [SRC])
    all_uses = external_uses(names, [SRC, TESTS])
    # Intra-module callers count here: a helper used only by its own module is
    # live production code, and excluding it would flag most of the codebase.
    any_src_uses = external_uses(names, [SRC], same_file=True)

    unreachable = []
    test_only = []
    for name, (path, kind, is_command) in sorted(DEFS.items()):
        if is_command or name in ALLOWED:
            continue
        if not all_uses[name]:
            unreachable.append((name, path, kind))
        elif not any_src_uses[name]:
            # Referenced from tests and nowhere else. `sample_for_second_pass`
            # was exactly this: a flat sampler superseded by a stratified one,
            # kept alive by its own test while the CLI used the replacement --
            # so the audit called it reachable and it was dead.
            test_only.append((name, path, kind))

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

    stubs = unimplemented_methods()
    if stubs:
        print(f"{len(stubs)} concrete method(s) that only raise NotImplementedError:")
        for name, path, line in stubs:
            print(f"  {name:<44} {path}:{line}")
        print()
        print("On a Protocol these are legitimate. On a concrete class they are a")
        print("gap: LlamaCppProvider.answer deferred to a phase that had already")
        print("shipped, so the demo could only ever run the extractive fallback.")
        print()

    if test_only:
        print(f"{len(test_only)} definition(s) referenced only by tests:")
        for name, path, kind in test_only:
            print(f"  {kind:<11} {name:<32} {path}")
        print()
        print("A function whose only caller is its own test is dead production")
        print("code with a passing test attached. Delete it, wire it, or allow-list it.")
        print()

    if not unreachable:
        if not test_only:
            print("No unexpected unreachable definitions.")
        return 1 if (stale or stubs or test_only) else 0

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
