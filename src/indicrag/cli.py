"""Command-line surface.

One Typer app with nested sub-apps, mirroring docs/ARCHITECTURE.md §18.

Formatting functions return `list[str]` rather than printing. The console path
and the `--report <path>` path then share one code path, which means what is
committed under `evals/` is byte-identical to what was shown on screen, and the
formatters are unit-testable without capturing stdout.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .config import get_settings
from .models import Document, Passage, QAItem, read_jsonl, write_jsonl

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
corpus_app = typer.Typer(no_args_is_help=True, help="Fetch, extract and segment the corpus.")
index_app = typer.Typer(no_args_is_help=True, help="Build lexical and dense indices.")
dataset_app = typer.Typer(no_args_is_help=True, help="Generate, verify and split the QA set.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")
app.add_typer(dataset_app, name="dataset")

GOLD_PATH = Path("evals/gold.jsonl")


def _emit(lines: list[str], report: Path | None) -> None:
    for line in lines:
        typer.echo(line)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        typer.echo(f"\nwritten to {report}")


def _stratified(items: list, n: int, *, seed: int = 20260922) -> list:
    """A seeded sample of `n` items spread evenly over the language pairs.

    Round-robins across the cells rather than taking a proportional slice of
    each, so a short sample still reaches every pair: the cross-lingual cells
    are the smallest and the ones the system exists to be measured on, and
    proportional rounding is exactly what drops them first.
    """
    import random
    from collections import defaultdict

    if n <= 0 or n >= len(items):
        return list(items)

    cells: dict[tuple[str, str], list] = defaultdict(list)
    for item in items:
        cells[(item.query_lang, item.passage_lang)].append(item)

    rng = random.Random(seed)
    for group in cells.values():
        group.sort(key=lambda i: i.id)
        rng.shuffle(group)

    out: list = []
    order = sorted(cells)
    while len(out) < n and any(cells[key] for key in order):
        for key in order:
            if cells[key] and len(out) < n:
                out.append(cells[key].pop())
    return sorted(out, key=lambda i: i.id)


# --- corpus --------------------------------------------------------------------


@corpus_app.command("fetch")
def corpus_fetch() -> None:
    """Resolve EN/HI article pairs, download both sides, write the manifest."""
    from .corpus.fetch import fetch_corpus
    from .corpus.sources import SCHEMES

    cfg = get_settings()
    result = fetch_corpus(SCHEMES, cfg.text_dir, progress=typer.echo)
    write_jsonl(cfg.manifest_path, result.documents)
    typer.echo("")
    typer.echo(result.summary())
    typer.echo(f"manifest -> {cfg.manifest_path}")


@corpus_app.command("validate")
def corpus_validate() -> None:
    """Run Devanagari integrity checks over every fetched Hindi document."""
    from .corpus import devanagari as dv

    cfg = get_settings()
    docs = list(read_jsonl(cfg.manifest_path, Document))
    if not docs:
        raise typer.BadParameter(f"no manifest at {cfg.manifest_path}; run `corpus fetch` first")

    failures = 0
    for doc in docs:
        path = cfg.text_dir / f"{doc.doc_id}.txt"
        if not path.exists():
            typer.echo(f"  MISSING {doc.doc_id}")
            failures += 1
            continue
        report = dv.validate(path.read_text(encoding="utf-8"), expect_hindi=(doc.lang == "hi"))
        if not report.ok:
            failures += 1
        typer.echo(f"  {doc.doc_id:28s} {report.summary()}")

    typer.echo("")
    typer.echo(f"{len(docs) - failures}/{len(docs)} documents pass integrity validation")
    if failures:
        raise typer.Exit(code=1)


@corpus_app.command("repair-spans")
def corpus_repair_spans(
    apply: bool = typer.Option(False, "--apply", help="write the repaired passages"),
) -> None:
    """Re-derive char_span and text_raw by aligning passages to their sources.

    Safe against a frozen corpus: `passage_id` and `text` are not touched, so no
    index, cache, gold reference or published result moves. Refuses to write if
    any of them does. Dry run by default.
    """
    from .corpus.align import format_repair, repair_spans
    from .corpus.integrity import audit, format_audit, load_sources

    cfg = get_settings()
    docs = list(read_jsonl(cfg.manifest_path, Document))
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter(f"no passages at {cfg.passages_path}; run `corpus segment` first")

    sources = load_sources(cfg.text_dir, docs)
    frozen = [(p.passage_id, p.text, p.token_count) for p in passages]

    result = repair_spans(passages, sources)
    for line in format_repair(result):
        typer.echo(line)

    typer.echo("")
    for line in format_audit(audit(passages, sources)):
        typer.echo(line)

    moved = [
        a for (a, b, c), p in zip(frozen, passages, strict=True)
        if (a, b, c) != (p.passage_id, p.text, p.token_count)
    ]
    if moved:
        typer.echo(f"\nREFUSING TO WRITE: {len(moved)} passages changed identity or text")
        raise typer.Exit(code=1)
    typer.echo(f"\nidentity check: passage_id, text and token_count unchanged "
               f"for all {len(passages)} passages")

    if apply:
        n = write_jsonl(cfg.passages_path, passages)
        typer.echo(f"{n} passages -> {cfg.passages_path}")
    else:
        typer.echo("dry run; pass --apply to write")


@corpus_app.command("audit")
def corpus_audit(report: Path | None = typer.Option(None, help="also write the audit here")) -> None:
    """Check every passage-level invariant the codebase claims.

    Separate from `corpus validate`, which reads the extracted *text*. This reads
    the *passages*, and it is the only thing that looks at `char_span` at all.
    """
    from .corpus.integrity import audit, format_audit, load_sources

    cfg = get_settings()
    docs = list(read_jsonl(cfg.manifest_path, Document))
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter(f"no passages at {cfg.passages_path}; run `corpus segment` first")

    result = audit(passages, load_sources(cfg.text_dir, docs))
    lines = format_audit(result)
    for line in lines:
        typer.echo(line)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        typer.echo(f"\nreport -> {report}")
    if result.blocking_failures:
        raise typer.Exit(code=1)


@corpus_app.command("segment")
def corpus_segment(
    version: int = typer.Option(
        1, "--version",
        help="Segmentation version. 1 is the frozen corpus every report and the gold "
        "set refer to; 2 applies the later fixes and re-points 312 passage IDs.",
    ),
) -> None:
    """Segment every document into passages with stable IDs.

    Defaults to the frozen version, because that is the corpus the gold set's
    passage IDs name. Running the corrected segmenter by default -- which is
    what this command did -- silently re-points a third of the gold set at
    different text, with no broken reference to show it (PLAN §10.1a).
    """
    from .corpus.segment import CURRENT_VERSION, FROZEN_VERSION, segment_document

    if version not in (FROZEN_VERSION, CURRENT_VERSION):
        raise typer.BadParameter(f"--version must be {FROZEN_VERSION} or {CURRENT_VERSION}")
    if version != FROZEN_VERSION:
        typer.echo(
            f"WARNING: segmentation v{version} is not the frozen corpus. Passage IDs it "
            "writes do not match evals/gold.jsonl; re-anchor the gold set before use."
        )

    cfg = get_settings()
    docs = list(read_jsonl(cfg.manifest_path, Document))
    if not docs:
        raise typer.BadParameter(f"no manifest at {cfg.manifest_path}; run `corpus fetch` first")

    passages: list[Passage] = []
    for doc in docs:
        path = cfg.text_dir / f"{doc.doc_id}.txt"
        if not path.exists():
            continue
        got = segment_document(doc, path.read_text(encoding="utf-8"), version=version)
        passages.extend(got)
        typer.echo(f"  {doc.doc_id:28s} {len(got):4d} passages")

    n = write_jsonl(cfg.passages_path, passages)
    typer.echo("")
    typer.echo(f"{n} passages (segmentation v{version}) -> {cfg.passages_path}")


@corpus_app.command("stats")
def corpus_stats(report: Path = typer.Option(None, "--report")) -> None:
    """Counts by scheme and language, plus a token-length histogram."""
    from .evaluation.report import format_corpus_stats

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter(f"no passages at {cfg.passages_path}; run `corpus segment` first")
    _emit(format_corpus_stats(passages), report)


# --- index ---------------------------------------------------------------------


@index_app.command("lexical")
def index_lexical() -> None:
    """Fit the TF-IDF and BM25 indices over the segmented corpus."""
    from .index.lexical import build_lexical

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")
    build_lexical(passages, cfg.lex_dir)
    typer.echo(f"lexical indices over {len(passages)} passages -> {cfg.lex_dir}")


@index_app.command("build")
def index_build(
    encoder: str = typer.Option("", "--encoder", help="Registry name; omit with --all."),
    build_all: bool = typer.Option(False, "--all", help="Every encoder in the registry."),
    pooling: str = typer.Option("", "--pooling", help="Override pooling (mean|cls)."),
    batch_size: int = typer.Option(16, "--batch-size"),
) -> None:
    """Encode the corpus with one dense encoder and cache the matrix.

    ARCHITECTURE §18 promised this command and it did not exist; the embeddings
    under `data/emb/` were built by an ad-hoc script. That is an NFR-6 problem
    rather than a cosmetic one -- an artefact every retrieval number depends on
    could not be regenerated by any committed command.
    """
    from .index.dense import build_dense
    from .index.encoders import REGISTRY

    if not encoder and not build_all:
        raise typer.BadParameter("pass --encoder <name> or --all")

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")

    names = list(REGISTRY) if build_all else [encoder]
    for name in names:
        try:
            _index, seconds = build_dense(
                passages, name, cfg.emb_dir,
                pooling=pooling or None,
                batch_size=batch_size,
                progress=typer.echo,
            )
            rate = len(passages) / seconds if seconds else 0.0
            typer.echo(f"  {name}: {len(passages)} passages in {seconds:.1f}s ({rate:.1f}/s)")
        except Exception as exc:  # noqa: BLE001 -- one gated model must not stop the rest
            typer.echo(f"  {name}: SKIPPED ({type(exc).__name__}: {exc})")


# --- dataset -------------------------------------------------------------------


@dataset_app.command("generate")
def dataset_generate(
    out: Path = typer.Option(GOLD_PATH, "--out"),
    gguf: str = typer.Option("", "--gguf", help="Path to a GGUF model file (llama.cpp)."),
    hf_model: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct", "--hf-model", help="transformers model id (default backend)."
    ),
    limit_per_cell: int = typer.Option(0, "--limit-per-cell", help="0 uses the PRD quotas."),
    seed: int = typer.Option(20260922, "--seed"),
) -> None:
    """Bootstrap QA candidates against the PRD §6.2 matrix. Writes verified=false."""
    from .dataset.generate import MATRIX, generate_candidates

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")

    model_path = gguf or cfg.llm_gguf_path
    if model_path:
        from .rag.providers import LlamaCppProvider

        provider = LlamaCppProvider(model_path, n_ctx=cfg.llm_n_ctx, n_threads=cfg.llm_n_threads)
    else:
        from .rag.providers import TransformersProvider

        provider = TransformersProvider(hf_model, threads=cfg.llm_n_threads)

    matrix = {k: limit_per_cell for k in MATRIX} if limit_per_cell else None
    items = generate_candidates(
        passages,
        provider.complete,
        seed=seed,
        matrix=matrix,
        progress=typer.echo,
        # A full run is hours of CPU inference; checkpoint so a crash costs one
        # item rather than the whole pass.
        checkpoint=lambda so_far: write_jsonl(out, so_far),
    )
    n = write_jsonl(out, items)
    typer.echo("")
    typer.echo(f"{n} candidates -> {out}  (all verified=false; run `dataset verify` next)")
    rate = getattr(provider, "parse_failure_rate", None)
    if rate is not None:
        typer.echo(
            f"model calls={provider.calls} retries={provider.retries} "
            f"parse failures={provider.parse_failures} ({rate:.1%})"
        )


@dataset_app.command("scaffold-unanswerable")
def dataset_scaffold_unanswerable(
    out: Path = typer.Option(Path("evals/unanswerable-scaffold.jsonl"), "--out"),
    merge_into: Path = typer.Option(None, "--merge-into", help="Append to an existing set."),
    seed: int = typer.Option(20260922, "--seed"),
) -> None:
    """Scaffold the 80 UNANSWERABLE items across the four PRD §6.3 classes.

    Not model-generated: a model asked for an unanswerable question returns the
    trivial out-of-scope kind, and the near-miss class is the one that actually
    tests hallucination.
    """
    from .dataset.unanswerable import build_unanswerable
    from .evaluation.run import title_map

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")

    items = build_unanswerable(passages, titles=title_map(cfg.manifest_path), seed=seed)

    if merge_into:
        existing = [i for i in read_jsonl(merge_into, QAItem) if i.answerable]
        write_jsonl(merge_into, existing + items)
        typer.echo(f"{len(items)} unanswerable + {len(existing)} answerable -> {merge_into}")
    else:
        write_jsonl(out, items)
        typer.echo(f"{len(items)} scaffolds -> {out}")
    typer.echo("All verified=false. Near-miss items carry a distractor passage to check against.")


@dataset_app.command("merge")
def dataset_merge(
    out: Path = typer.Option(GOLD_PATH, "--out"),
    sources: str = typer.Option(
        "evals/gold.jsonl,evals/answerable-candidates.jsonl,evals/hindi-candidates.jsonl,"
        "evals/hindi-candidates-2.jsonl,evals/hindi-candidates-3.jsonl,"
        "evals/hindi-candidates-4.jsonl",
        "--sources",
        help="Comma-separated checkpoint files, in precedence order.",
    ),
) -> None:
    """Consolidate generation checkpoints into the gold set.

    Generation checkpoints after every item but only merges into gold.jsonl when
    a run completes. An interrupted run therefore leaves its work sitting in the
    checkpoint files, invisible to everything downstream. This recovers it.

    Safe to run repeatedly. Unanswerable items already in the target are
    preserved -- they come from `scaffold-unanswerable`, not from generation, and
    re-merging must not drop them.

    Deduplication is on the gold passage set, NOT on the item id. Ids are
    numbered per cell within a run, so a resumed run reissues `qa-hinglish-hi-000`
    and an id-keyed merge silently drops the earlier item. The gold passage set is
    unique per item by construction -- no passage is used twice -- so it is the
    reliable key. Ids are renumbered on output to restore uniqueness.
    """
    seen: set[tuple[str, ...]] = set()
    answerable: list[QAItem] = []
    for name in [part.strip() for part in sources.split(",") if part.strip()]:
        path = Path(name)
        if not path.exists():
            typer.echo(f"  (absent, skipped) {name}")
            continue
        kept = dropped = 0
        for item in read_jsonl(path, QAItem):
            if not item.answerable:
                continue
            key = tuple(sorted(item.gold_passage_ids))
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            answerable.append(item)
            kept += 1
        note = f" ({dropped} duplicate passages skipped)" if dropped else ""
        typer.echo(f"  {kept:4d} answerable from {name}{note}")

    # Renumber so ids are unique across the merged set.
    counters: dict[str, int] = {}
    for item in answerable:
        cell = f"{item.query_lang}-{item.passage_lang}"
        n = counters.get(cell, 0)
        counters[cell] = n + 1
        item.id = f"qa-{cell}-{n:03d}"

    existing = list(read_jsonl(out, QAItem)) if out.exists() else []
    unanswerable = [i for i in existing if not i.answerable]
    write_jsonl(out, answerable + unanswerable)
    typer.echo("")
    typer.echo(
        f"{len(answerable)} answerable + {len(unanswerable)} unanswerable "
        f"= {len(answerable) + len(unanswerable)} -> {out}"
    )
    typer.echo("Run `indicrag dataset stats` to see coverage against the PRD matrix.")


@dataset_app.command("verify")
def dataset_verify(
    path: Path = typer.Option(GOLD_PATH, "--path"),
    annotator: str = typer.Option("a1", "--annotator"),
    limit: int = typer.Option(0, "--limit", help="Review at most N items this sitting."),
    assist: Path = typer.Option(
        Path("evals/review-assist.jsonl"), "--assist",
        help="Pre-review notes from `dataset review`, shown under each item if present.",
    ),
    recheck: bool = typer.Option(
        False, "--recheck",
        help="Review items a model verified; each one accepted is recorded under --annotator.",
    ),
    split: str = typer.Option("", "--split", help="Only items in this split (dev or test)."),
) -> None:
    """Interactive review. Saves after every decision; resumable."""
    import json as _json

    from .dataset.verify import verify_loop

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not path.exists():
        raise typer.BadParameter(f"no candidates at {path}; run `dataset generate` first")

    notes: dict[str, dict] = {}
    if assist.exists():
        with assist.open(encoding="utf-8") as fh:
            notes = {r["id"]: r for r in map(_json.loads, fh) if r.get("id")}
        typer.echo(f"pre-review notes for {len(notes)} items from {assist} (advice only)")

    progress = verify_loop(
        path,
        passages,
        annotator=annotator,
        ask=typer.prompt,
        say=typer.echo,
        limit=limit or None,
        assist=notes,
        recheck=recheck,
        split=split or None,
    )
    typer.echo("")
    typer.echo(str(progress))


@dataset_app.command("second-pass")
def dataset_second_pass(
    path: Path = typer.Option(GOLD_PATH, "--path"),
    sample_out: Path = typer.Option(Path("evals/second-pass.jsonl"), "--sample"),
    compare_with: Path = typer.Option(None, "--compare", help="Score a filled-in sample."),
    fraction: float = typer.Option(0.15, "--fraction"),
    seed: int = typer.Option(20260922, "--seed"),
    report: Path = typer.Option(None, "--report"),
    label: Path = typer.Option(
        None, "--label", help="Label a blind sample interactively (question and evidence only)."
    ),
    labeller: str = typer.Option("", "--labeller", help="Name recorded on each label."),
) -> None:
    """Draw a blind re-labelling sample, or score one and report Cohen's kappa.

    Without --compare, writes a stratified sample with the first pass's verdict
    stripped out. Fill in each `answerable` field independently, then re-run with
    --compare pointing at the filled file.

    PRD §10.2 gates Module 5 on kappa >= 0.70.
    """
    from .dataset.second_pass import (
        EVIDENCE_SIZE,
        compare,
        draw_sample,
        format_agreement,
        write_blind_sample,
    )

    if label is not None:
        from .dataset.second_pass import label_blind

        if not labeller:
            raise typer.BadParameter("--label needs --labeller <name>")
        cfg = get_settings()
        done, total = label_blind(
            label, list(read_jsonl(cfg.passages_path, Passage)),
            labeller=labeller, ask=typer.prompt, say=typer.echo,
        )
        typer.echo(f"\n{done}/{total} labelled -> {label}")
        if done == total:
            typer.echo(f"then: indicrag dataset second-pass --compare {label}")
        return

    items = list(read_jsonl(path, QAItem))
    if not items:
        raise typer.BadParameter(f"no items at {path}")

    if compare_with is not None:
        _emit(format_agreement(compare(items, compare_with)), report)
        return

    sample = draw_sample(items, fraction=fraction, seed=seed)
    if not sample:
        raise typer.BadParameter(
            "no verified items to sample; run `dataset verify` first -- "
            "agreement on unverified labels would measure nothing"
        )
    # Evidence for every sampled item comes from retrieval, so an unanswerable
    # item is not recognisable by having none (see second_pass.blind).
    from .index.dense import DenseIndex, Encoder
    from .index.encoders import get
    from .index.hybrid import rrf_fusion
    from .index.lexical import LexicalIndex

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    lex = LexicalIndex.load(cfg.lex_dir)
    spec = get(cfg.encoder_primary)
    dense = DenseIndex.load(spec, cfg.emb_dir, passages)
    encoder = Encoder(spec)
    retrieved = {
        i.id: [
            r.passage_id
            for r in rrf_fusion(
                lex.search_bm25(i.question, 50),
                dense.search_vector(encoder.encode_query(i.question), 50),
                k=EVIDENCE_SIZE,
            )
        ]
        for i in sample
    }
    n = write_blind_sample(sample, sample_out, retrieved)
    from collections import Counter

    strata = Counter(
        "answerable" if i.answerable else (i.unanswerable_class or "other") for i in sample
    )
    typer.echo(f"{n} items -> {sample_out}")
    typer.echo(f"  strata: {dict(strata)}")
    typer.echo("")
    typer.echo("Label it WITHOUT consulting the first pass:")
    typer.echo(f"  indicrag dataset second-pass --label {sample_out} --labeller <name>")
    typer.echo(f"then: indicrag dataset second-pass --compare {sample_out}")


@dataset_app.command("review")
def dataset_review(
    gold: Path = typer.Option(GOLD_PATH, "--gold"),
    notes: Path = typer.Option(
        None, "--notes", help="JSONL from a reading pass: id, verdict, issues, suggestions."
    ),
    reviewed_by: str = typer.Option("", "--reviewed-by", help="Who wrote --notes. Required with it."),
    out: Path = typer.Option(Path("evals/review-assist.jsonl"), "--out"),
    report: Path = typer.Option(None, "--report"),
) -> None:
    """Pre-review the gold set for `dataset verify`. Advice only; never sets `verified`.

    Runs the mechanical checks in dataset/review.py over every item and, with
    --notes, merges a reading pass's per-item notes. `dataset verify` shows the
    result under each item's evidence.
    """
    import json as _json

    from .dataset.review import format_review, merge, review

    if notes is not None and not reviewed_by:
        raise typer.BadParameter("--notes needs --reviewed-by: advice must say whose it is")

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")
    items = list(read_jsonl(gold, QAItem))

    by_id: dict[str, dict] = {}
    if notes is not None:
        with notes.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = _json.loads(line)
                    by_id[rec["id"]] = rec
        unknown = sorted(set(by_id) - {i.id for i in items})
        if unknown:
            raise typer.BadParameter(f"notes for items not in {gold}: {', '.join(unknown[:5])}")

    records = merge(review(items, passages), by_id, reviewed_by=reviewed_by)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    _emit(format_review(records, items) + ["", f"{len(records)} records -> {out}"], report)


@dataset_app.command("reanchor")
def dataset_reanchor(
    out: Path = typer.Option(..., "--out", help="Proposed gold file. Never evals/gold.jsonl."),
    gold: Path = typer.Option(GOLD_PATH, "--gold"),
    to_version: int = typer.Option(2, "--to-version"),
    report: Path = typer.Option(None, "--report"),
) -> None:
    """Map gold citations from the frozen segmentation to a newer one.

    Both versions are segmented here from the fetched text, so the answer does
    not depend on which passages.jsonl happens to be on disk. Writes a proposed
    gold file and never the gold set itself: adopting it is a verification
    decision, not a build step.
    """
    from .corpus.segment import FROZEN_VERSION, segment_document
    from .dataset.reanchor import apply, format_reanchor, reanchor

    if out.resolve() == gold.resolve():
        raise typer.BadParameter("--out must not be the gold set itself")
    if to_version == FROZEN_VERSION:
        raise typer.BadParameter(f"--to-version must differ from the frozen v{FROZEN_VERSION}")

    cfg = get_settings()
    docs = list(read_jsonl(cfg.manifest_path, Document))
    old: list[Passage] = []
    new: list[Passage] = []
    for doc in docs:
        path = cfg.text_dir / f"{doc.doc_id}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        old.extend(segment_document(doc, text, version=FROZEN_VERSION))
        new.extend(segment_document(doc, text, version=to_version))
    if not old:
        raise typer.BadParameter(f"no fetched text under {cfg.text_dir}; run `corpus fetch` first")

    items = list(read_jsonl(gold, QAItem))
    result = reanchor(items, old, new)
    n = write_jsonl(out, apply(items, result, version=to_version))
    lines = [f"v{FROZEN_VERSION}: {len(old)} passages   v{to_version}: {len(new)} passages", ""]
    lines += format_reanchor(result, old_version=FROZEN_VERSION, new_version=to_version)
    lines += ["", f"{n} items -> {out}  (proposed; {gold} is unchanged)"]
    _emit(lines, report)


@dataset_app.command("stats")
def dataset_stats(
    path: Path = typer.Option(GOLD_PATH, "--path"),
    report: Path = typer.Option(None, "--report"),
) -> None:
    """Coverage against the PRD matrix. Run this during annotation, not after."""
    from .dataset.split import (
        answer_provenance,
        coverage,
        format_coverage,
        format_provenance,
        format_unknown_fields,
    )
    from .dataset.verify import progress_of

    items = list(read_jsonl(path, QAItem))
    if not items:
        raise typer.BadParameter(f"no items at {path}")

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    lines = format_coverage(coverage(items))
    lines += format_unknown_fields(items)
    if passages:
        lines += format_provenance(answer_provenance(items, passages))
    lines += ["", str(progress_of(items))]
    _emit(lines, report)


@dataset_app.command("split")
def dataset_split(
    path: Path = typer.Option(GOLD_PATH, "--path"),
    dev_size: int = typer.Option(120, "--dev-size"),
    seed: int = typer.Option(20260922, "--seed"),
) -> None:
    """Stratified dev/test split over VERIFIED items only. Seals the test split."""
    import json

    from .dataset.split import stratified_split

    items = list(read_jsonl(path, QAItem))
    dev, test = stratified_split(items, dev_size=dev_size, seed=seed)
    if not dev and not test:
        raise typer.BadParameter("no verified items to split; run `dataset verify` first")

    write_jsonl(path, items)
    Path("evals/splits.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "dev": sorted(i.id for i in dev),
                "test": sorted(i.id for i in test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"dev={len(dev)} test={len(test)} -> evals/splits.json")
    typer.echo("The test split is now sealed: tune only on dev until P5.")


eval_app = typer.Typer(no_args_is_help=True, help="Run and report the evaluations.")
app.add_typer(eval_app, name="eval")


@eval_app.command("all")
def eval_all(
    out_dir: Path = typer.Option(Path("evals"), "--report", help="Directory for reports."),
    gold: Path = typer.Option(GOLD_PATH, "--gold"),
) -> None:
    """Regenerate every committed report from cached artefacts (PRD NFR-6).

    Each stage degrades rather than fails: a stage whose inputs are missing is
    skipped with a reason and the rest still run.
    """
    from .evaluation.all_reports import run_all

    summary = run_all(out_dir, gold_path=gold, progress=lambda m: typer.echo(f"  {m}..."))
    for line in summary.format():
        typer.echo(line)


@eval_app.command("retrieval")
def eval_retrieval(
    gold: Path = typer.Option(GOLD_PATH, "--gold"),
    report: Path = typer.Option(None, "--report"),
    alpha: float = typer.Option(0.4, "--alpha", help="Weighted-fusion mixing weight."),
    sweep: bool = typer.Option(
        True, "--sweep/--no-sweep", help="Also sweep alpha end to end (H2)."
    ),
    split: str = typer.Option("all", "--split", help="all | dev | test (after `dataset split`)."),
) -> None:
    """Module 1-3: lexical, dense and fusion retrieval, with paired tests."""
    from .evaluation.all_reports import provenance
    from .evaluation.report import format_by_slice, format_paired, format_retrieval
    from .evaluation.run import alpha_sweep, run_retrieval

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")

    if split not in {"all", "dev", "test"}:
        raise typer.BadParameter("--split must be all, dev or test")
    items = [
        i for i in read_jsonl(gold, QAItem)
        if i.answerable and i.gold_passage_ids and (split == "all" or i.split == split)
    ]
    source = str(gold) + ("" if split == "all" else f" [{split} split]")
    if not items and split != "all":
        raise typer.BadParameter(f"no answerable items in the {split} split; run `dataset split`")
    if not items:
        items = list(read_jsonl(Path("evals/probes.jsonl"), QAItem))
        source = "synthetic probes"
    if not items:
        raise typer.BadParameter("no answerable items and no probes to fall back on")

    reports, skipped = run_retrieval(passages, items, alpha=alpha, progress=typer.echo)
    lines = provenance(items, source)
    lines += format_retrieval(reports) + [""] + format_by_slice(reports)
    lines += [""] + format_paired(reports, baseline="BM25")
    if skipped:
        lines += ["", "NOT EVALUATED", "-" * 78] + [f"  {s}" for s in skipped]

    if sweep:
        # The alpha sweep produced a number the paper quotes and a figure it
        # plots, and until now no command produced it -- it existed only in
        # evals/report-retrieval-probes.txt, from a script that is not in the
        # repository. NFR-6 says every reported table regenerates from one
        # command; this is one of the tables.
        typer.echo("  alpha sweep")
        swept = alpha_sweep(passages, items)
        if swept:
            lines += ["", "ALPHA SWEEP -- weighted fusion, Recall@5", "-" * 78, ""]
            lines += [
                f"  BM25 + {swept.dense}. alpha=1.0 pure lexical, alpha=0.0 pure dense.",
                "",
            ]
            lines += [f"  a={a:.1f}  {r:.3f}" for a, r in swept.points]
            lines += ["", f"  {swept.verdict()}"]
        else:
            lines += ["", "ALPHA SWEEP -- skipped: no dense index available."]

    _emit(lines, report)


@eval_app.command("probes")
def eval_probes(
    out: Path = typer.Option(Path("evals/probes.jsonl"), "--out"),
    per_shape: int = typer.Option(60, "--per-shape"),
) -> None:
    """Rebuild the synthetic probe set used when no verified gold set exists.

    Every preliminary retrieval number in this repository is measured on these,
    and nothing regenerated them: `evals/probes.jsonl` was committed output from
    a script that is not. A probe set that cannot be rebuilt cannot be checked.
    """
    from .evaluation.run import load_probes

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")

    probes = load_probes(passages, per_shape=per_shape)
    n = write_jsonl(out, probes)
    from collections import Counter

    shapes = Counter(p.notes.split()[0] if p.notes else "?" for p in probes)
    typer.echo(f"{n} probes -> {out}")
    for shape, count in sorted(shapes.items()):
        typer.echo(f"  {shape:<16}{count:>5}")


@eval_app.command("answerability")
def eval_answerability(
    gold: Path = typer.Option(GOLD_PATH, "--gold"),
    report: Path = typer.Option(None, "--report"),
    method: str = typer.Option("hybrid", "--method"),
    k: int = typer.Option(5, "--k"),
    dev_fraction: float = typer.Option(0.3, "--dev-fraction", help="Share used to fit tau."),
    gguf: str = typer.Option(
        "", "--gguf",
        help="Also evaluate the generator self-report signal. Needs a generation per item.",
    ),
    sample: int = typer.Option(
        0, "--sample", help="Evaluate a seeded sample instead of every item."
    ),
    enrich: bool = typer.Option(
        False, "--enrich",
        help="With --sample: keep every unanswerable item, so the per-class table is readable.",
    ),
) -> None:
    """Module 5: can the system tell an answerable question from an unanswerable one?

    The per-class recall table is the result, not the aggregate F1 -- an
    aggregate can look respectable while the system fails completely on the
    classes that matter, and it can look poor while the system is merely
    abstaining on everything.
    """
    from .evaluation.all_reports import provenance
    from .evaluation.answerability import (
        enriched_sample,
        feature_separation,
        format_answerability,
        format_calibrated,
        format_feature_separation,
        format_separation,
        format_separation_across_methods,
        format_tau_sweep,
        run_answerability,
        run_calibrated,
        separation_across_methods,
        stratified_by_class,
        tau_sweep,
    )

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    items = list(read_jsonl(gold, QAItem))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")
    if not any(not i.answerable for i in items):
        raise typer.BadParameter(
            f"{gold} has no unanswerable items; answerability cannot be measured "
            "against a set that is entirely answerable"
        )

    if sample:
        items = (enriched_sample if enrich else stratified_by_class)(items, sample)
        how = "every unanswerable item plus answerable fill" if enrich else "stratified by class"
        typer.echo(f"sampling {len(items)} items, {how}")

    result, meta = run_answerability(
        passages, items, method=method, k=k,
        dev_fraction=dev_fraction, progress=typer.echo,
    )

    lines = provenance(items, str(gold))
    if sample and enrich:
        n_un = sum(1 for i in items if not i.answerable)
        lines += [
            "ENRICHED SAMPLE. Every unanswerable item is kept and the rest filled",
            f"with answerable ones, giving a base rate of {n_un / len(items):.3f} against",
            "0.200 on the full set. This is what makes the per-class table readable",
            "at a tenth of the generation cost, and it means PRECISION AND F1 HERE",
            "ARE NOT COMPARABLE to a proportional run -- a threshold abstaining at",
            "chance scores its precision at the base rate, and the base rate moved.",
            "Recall and the per-class breakdown are unaffected.",
            "",
        ]
    lines += [
        f"tau fitted on {len(meta['dev'])} items (F1={meta['dev_f1']:.3f}); "
        f"reported on {len(meta['test'])} held out.",
        "",
    ]
    lines += format_answerability(result)
    lines += [""] + format_separation(meta["scores"], meta["labels"], method=method)
    lines += [""] + format_separation_across_methods(
        separation_across_methods(passages, items, k=k)
    )
    lines += [""] + format_feature_separation(
        feature_separation(meta["features"], meta["labels"]), method=method
    )
    lines += [""] + format_tau_sweep(tau_sweep(meta["features"], meta["labels"]))

    sep = feature_separation(meta["features"], meta["labels"])
    cal_report, cal_meta = run_calibrated(
        meta["features"], meta["labels"], meta["dev"], meta["test"], items=items
    )
    lines += ["", "=" * 78, ""] + format_calibrated(
        cal_report, cal_meta,
        best_single_auc=max((abs(v[2] - 0.5) for v in sep.values()), default=0.0) + 0.5,
    )
    lines += [""] + format_answerability(cal_report)

    if gguf:
        lines += ["", "=" * 78, ""] + _generator_signals(
            items, passages, meta["hits"], meta["labels"],
            meta["dev"], meta["test"], gguf=gguf, k=k,
        )

    _emit(lines, report)


def _generator_signals(items, passages, hits, labels, dev, test, *, gguf: str, k: int) -> list:
    """ARCHITECTURE §12 signals 2 and 3, evaluated beside the retrieval ones.

    These are the two signals that read the passage *text*. Signals 1 and 4 see
    only score geometry, and §12.4 records that on this corpus that geometry is
    near chance, because 55 of the 80 unanswerable items are about schemes that
    are present. A question about a real scheme retrieves like any other; the
    difference is whether the specific fact is in the passage, and only a reader
    can tell.

    Both are fitted on the same split as the threshold so all four are
    comparable, and both reuse one generation per item.
    """
    from .answerability.nli import NLIScorer
    from .answerability.signals import (
        EntailmentSignal,
        SelfReport,
        SelfReportSignal,
        unanswerable_f1,
    )
    from .evaluation.answerability import (
        AnswerabilityOutcome,
        AnswerabilityReport,
        format_answerability,
    )
    from .query.langid import classify
    from .rag.arms import Generation, GenerationCache, parse_answer, prompt_hash
    from .rag.prompts import build_answer_prompt
    from .rag.providers import LlamaCppProvider

    cfg = get_settings()
    model = Path(gguf).stem
    by_id = {p.passage_id: p for p in passages}
    provider = LlamaCppProvider(gguf, n_ctx=cfg.llm_n_ctx, n_threads=cfg.llm_n_threads)
    # Its own cache file. These generations cover unanswerable items, which
    # Module 4 never sees, so sharing the arms cache would make that file's
    # contents depend on which command last wrote it.
    cache = GenerationCache(cfg.gen_dir / f"{model}-answerability.jsonl")

    reports: list[SelfReport] = []
    pairs: list[tuple[str, str]] = []

    typer.echo(f"  generating for {len(items)} items (this is the slow part)")
    for n, (item, hit) in enumerate(zip(items, hits, strict=True), start=1):
        context = [by_id[h.passage_id] for h in hit[:k] if h.passage_id in by_id]
        prompt = build_answer_prompt(item.question, context)
        key = prompt_hash(prompt, model, "answerability")

        gen = cache.get(key)
        if gen is None:
            data = parse_answer(provider.complete(prompt))
            answer = (data.get("answer") or "").strip() if data else ""
            gen = Generation(
                item_id=item.id,
                arm="S",
                question=item.question,
                answerable=bool(data and data.get("answerable", True)) and bool(answer),
                answer=answer,
                citation=(data.get("citation") or "").strip() if data else "",
                confidence=float(data.get("confidence") or 0.0) if data else 0.0,
                context_ids=[p.passage_id for p in context],
                model=model,
                prompt_hash=key,
            )
            cache.put(gen)
            if n % 25 == 0:
                typer.echo(f"    {n}/{len(items)}")

        reports.append(SelfReport.from_generation(gen))
        # Collected on the cache-hit path too. An earlier version built this
        # only when generating, so a resumed run scored entailment on whatever
        # subset happened to be new.
        cited = [by_id[pid].text for pid in gen.context_ids if pid in by_id]
        pairs.append((" ".join(cited), gen.answer))

    out: list[str] = []

    # --- signal 2 -----------------------------------------------------------
    signal, dev_f1 = SelfReportSignal.fit(
        [reports[n] for n in dev], [labels[n] for n in dev]
    )
    result = AnswerabilityReport(
        system=f"generator self-report (min_confidence={signal.min_confidence:.2f})"
    )
    for n in test:
        item = items[n]
        result.outcomes.append(
            AnswerabilityOutcome(
                item_id=item.id,
                gold_answerable=item.answerable,
                pred_answerable=signal.predict_answerable(reports[n]),
                query_type=classify(item.question).query_type,
                unanswerable_class=item.unanswerable_class or "",
            )
        )

    flag_only = SelfReportSignal(min_confidence=0.0)
    flag_f1 = unanswerable_f1(
        [flag_only.predict_answerable(reports[n]) for n in test], [labels[n] for n in test]
    )
    out += [
        f"min_confidence fitted on {len(dev)} items (F1={dev_f1:.3f}); "
        f"reported on {len(test)} held out.",
        f"The `answerable` flag alone, with no confidence gate, scores F1={flag_f1:.3f} "
        "on the same rows.",
        "",
        *format_answerability(result),
    ]

    # --- signal 3 -----------------------------------------------------------
    if not NLIScorer.available(cfg.nli_model):
        out += [
            "",
            "=" * 78,
            "",
            f"NLI signal skipped: {cfg.nli_model} is not present locally.",
        ]
        return out

    typer.echo(f"  scoring entailment for {len(pairs)} pairs")
    scorer = NLIScorer(
        cfg.nli_model, cache_path=cfg.gen_dir / "nli-cache.jsonl", threads=cfg.llm_n_threads
    )
    abstained = [not r.answerable for r in reports]
    entailments = [
        None if abstained[i] else v.entailment
        for i, v in enumerate(scorer.score_pairs(pairs))
    ]

    ent_signal, ent_dev_f1 = EntailmentSignal.fit(
        [entailments[n] for n in dev],
        [labels[n] for n in dev],
        abstentions=[abstained[n] for n in dev],
    )
    ent_result = AnswerabilityReport(system=f"NLI entailment (tau={ent_signal.tau:.3f})")
    for n in test:
        item = items[n]
        ent_result.outcomes.append(
            AnswerabilityOutcome(
                item_id=item.id,
                gold_answerable=item.answerable,
                pred_answerable=ent_signal.predict_answerable(
                    entailments[n], abstained=abstained[n]
                ),
                query_type=classify(item.question).query_type,
                unanswerable_class=item.unanswerable_class or "",
            )
        )

    out += [
        "",
        "=" * 78,
        "",
        f"tau fitted on {len(dev)} items (F1={ent_dev_f1:.3f}); "
        f"reported on {len(test)} held out.",
        "An abstention is UNANSWERABLE without consulting the score: there is no",
        "answer to entail, and scoring the empty string against a passage measures",
        "nothing.",
        "",
        *format_answerability(ent_result),
    ]
    return out


@eval_app.command("errors")
def eval_errors(
    gold: Path = typer.Option(GOLD_PATH, "--gold"),
    report: Path = typer.Option(None, "--report"),
    out: Path = typer.Option(Path("evals/errors.jsonl"), "--out"),
    limit: int = typer.Option(20, "--limit"),
    per_category: int = typer.Option(2, "--per-category"),
    split: str = typer.Option("all", "--split", help="all | dev | test (after `dataset split`)."),
    system: str = typer.Option(
        "Hybrid RRF (", "--system",
        help="Prefix of the system whose failures are analysed. The default is plain RRF, "
        "which the committed probe-era cases analyse; 'Hybrid RRF script-aware' is the "
        "system the paper recommends.",
    ),
) -> None:
    """Module 6: the difficult cases, selected by rule rather than by hand."""
    import json as _json

    from .evaluation.all_reports import provenance
    from .evaluation.errors import collect_cases, format_errors
    from .evaluation.run import run_retrieval

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if split not in {"all", "dev", "test"}:
        raise typer.BadParameter("--split must be all, dev or test")
    items = [
        i for i in read_jsonl(gold, QAItem)
        if i.answerable and i.gold_passage_ids and (split == "all" or i.split == split)
    ]
    source = str(gold) if split == "all" else f"{gold} [{split} split]"
    if not items and split == "all":
        items = list(read_jsonl(Path("evals/probes.jsonl"), QAItem))
        source = "evals/probes.jsonl"
    if not passages or not items:
        raise typer.BadParameter("need passages and answerable items")

    reports, _skipped = run_retrieval(passages, items, progress=typer.echo)
    by_name = {r.system: r for r in reports}
    primary = next((n for n in by_name if n.startswith(system)), None)
    if primary is None:
        # Without a dense index there is no hybrid to analyse; say so and fall
        # back to the first system rather than failing the whole report.
        primary = next(iter(by_name))
        typer.echo(f"no system starts with {system!r}; analysing {primary}")

    other = {
        name: {o.item_id: list(o.retrieved) for o in rep.outcomes}
        for name, rep in by_name.items()
        if name != primary
    }
    cases = collect_cases(
        by_name[primary].outcomes, items, passages,
        per_category=per_category, other_runs=other,
    )[:limit]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for case in cases:
            fh.write(_json.dumps(case.as_dict(), ensure_ascii=False) + "\n")
    typer.echo(f"{len(cases)} cases -> {out}")
    _emit(
        provenance(items, source) + [f"System analysed: {primary}", ""]
        + format_errors(cases, passages),
        report,
    )


@eval_app.command("qa")
def eval_qa(
    gold: Path = typer.Option(GOLD_PATH, "--gold"),
    report: Path = typer.Option(None, "--report"),
    gguf: str = typer.Option("", "--gguf", help="GGUF model; omit for the extractive baseline."),
    arms: str = typer.Option("A,B,C,D", "--arms"),
    k: int = typer.Option(5, "--k"),
    sample: int = typer.Option(
        0, "--sample",
        help="Run a seeded sample stratified by language pair instead of the full set.",
    ),
    no_model: bool = typer.Option(False, "--no-model", help="Extractive provider only."),
    nli: bool = typer.Option(
        False, "--nli",
        help="Verify citations by entailment as well as word overlap (~0.3s/answer).",
    ),
    split: str = typer.Option("all", "--split", help="all | dev | test (after `dataset split`)."),
) -> None:
    """Module 4: direct LLM vs retrieval-augmented question answering."""
    from .evaluation.qa_run import format_module4, run_module4
    from .index.dense import DenseIndex, Encoder
    from .index.encoders import get
    from .index.hybrid import script_aware_rrf
    from .index.lexical import LexicalIndex
    from .query.langid import classify
    from .rag.arms import GenerationCache

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if split not in {"all", "dev", "test"}:
        raise typer.BadParameter("--split must be all, dev or test")
    items = [
        i for i in read_jsonl(gold, QAItem)
        if i.answerable and (split == "all" or i.split == split)
    ]
    if not items:
        raise typer.BadParameter(f"no answerable items in {gold}")

    if sample:
        # Stratify by language pair, not a flat head(). Generation is the
        # expensive stage -- roughly 67s per item across four arms on this
        # machine -- so a partial run is the normal case rather than the
        # exception, and a flat prefix of a file grouped by cell would report
        # four arms measured only on en->en. The per-cell contrast is the whole
        # point of Module 4, so the sample has to preserve it.
        items = _stratified(items, sample)
        typer.echo(f"sampling {len(items)} items, stratified by language pair")

    verified = sum(1 for i in items if i.verified)
    if verified < len(items):
        typer.echo(
            f"WARNING: {len(items) - verified} of {len(items)} items are unverified. "
            "Results below are preliminary and must not be reported as gold-set numbers."
        )

    # Retrievers for arms B and C.
    retrievers: dict = {}
    lex = LexicalIndex.load(cfg.lex_dir)
    try:
        spec = get(cfg.encoder_primary)
        dense = DenseIndex.load(spec, cfg.emb_dir, passages)
        enc = Encoder(spec)
        script_of = {p.passage_id: ("deva" if p.lang == "hi" else "latin") for p in passages}
        qvec = {i.id: enc.encode_query(i.question) for i in items}
        retrievers["B"] = lambda i: [h.passage_id for h in dense.search_vector(qvec[i.id], k)]
        retrievers["C"] = lambda i: [
            h.passage_id
            for h in script_aware_rrf(
                lex.search_bm25(i.question, 50),
                dense.search_vector(qvec[i.id], 50),
                script_of=script_of,
                query_script=classify(i.question).script,
                k=k,
            )
        ]
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        typer.echo(f"dense index unavailable ({exc}); arms B and C will be skipped")

    if gguf and not no_model:
        from .rag.providers import LlamaCppProvider

        provider = LlamaCppProvider(gguf, n_ctx=cfg.llm_n_ctx, n_threads=cfg.llm_n_threads)
        complete = provider.complete
        model = Path(gguf).stem
    else:
        from .rag.extractive_json import extractive_complete

        complete = extractive_complete(passages)
        model = "extractive"

    scorer = None
    if nli:
        from .answerability.nli import NLIScorer

        if not NLIScorer.available(cfg.nli_model):
            raise typer.BadParameter(
                f"--nli needs {cfg.nli_model}, which is not present locally. "
                "Fetch it first, or drop --nli to report lexical support only."
            )
        scorer = NLIScorer(
            cfg.nli_model,
            cache_path=cfg.gen_dir / "nli-cache.jsonl",
            threads=cfg.llm_n_threads,
        )

    result = run_module4(
        items, passages, complete,
        retrievers=retrievers,
        cache=GenerationCache(cfg.gen_dir / f"{model}.jsonl"),
        model=model, k=k, nli=scorer,
        arms=[a.strip().upper() for a in arms.split(",") if a.strip()],
        progress=typer.echo,
    )
    from .evaluation.all_reports import provenance

    # The same banner as every other report, so a Module 4 figure copied out of
    # it keeps what it rests on: which split, how many items, verified by whom.
    source = str(gold) if split == "all" else f"{gold} [{split} split]"
    if sample:
        source += f", {len(items)}-item stratified sample"
    _emit(provenance(items, source) + format_module4(result), report)


@app.command("ask")
def ask(
    query: str = typer.Argument(..., help="The question, in English, Hindi or Hinglish."),
    k: int = typer.Option(5, "--k"),
    method: str = typer.Option(
        "hybrid", "--method",
        help="hybrid (script-aware, default) | hybrid-rrf | hybrid-weighted | bm25 | tfidf | dense",
    ),
    gguf: str = typer.Option(
        "", "--gguf",
        help="Generate the answer with this GGUF model instead of extracting a sentence.",
    ),
    bm25_floor: float = typer.Option(
        0.0, "--bm25-floor",
        help="Refuse when the BM25 top score is below this. 18.08 is the threshold fitted "
        "on the dev split (tau 0.25 x scale 72.31, paper §VI-H); 0 disables it.",
    ),
) -> None:
    """Answer one question and print the full response object."""
    from .pipeline import Retrievers, answer_query

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")

    # Load the dense index, not just the lexical one. Without it `hybrid` --
    # the default, and the method this project's central result is about --
    # degrades to plain BM25 inside `retrieve()`, silently. That degradation is
    # the right behaviour for a fresh checkout with no embeddings built, but it
    # made the demo answer every query with the one configuration the paper
    # shows collapses cross-lingually: BM25 scores 0.006 there against 0.153
    # for script-aware fusion. A demo that cannot exercise the contribution is
    # not a demo of this system.
    from .index.lexical import LexicalIndex

    retrievers = Retrievers(lexical=LexicalIndex.load(cfg.lex_dir))
    try:
        from .index.dense import DenseIndex, Encoder
        from .index.encoders import get

        spec = get(cfg.encoder_primary)
        retrievers.dense = DenseIndex.load(spec, cfg.emb_dir, passages)
        retrievers.encoder = Encoder(spec)
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        typer.echo(
            f"# dense index unavailable ({type(exc).__name__}: {exc});"
            f" '{method}' will fall back to lexical retrieval",
            err=True,
        )

    provider = None
    if gguf:
        from .rag.providers import LlamaCppProvider

        cfg_llm = get_settings()
        provider = LlamaCppProvider(
            gguf, n_ctx=cfg_llm.llm_n_ctx, n_threads=cfg_llm.llm_n_threads
        )

    result = answer_query(
        query, passages, retrievers=retrievers, method=method, k=k,
        bm25_floor=bm25_floor, provider=provider,
    )
    import json

    typer.echo(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
