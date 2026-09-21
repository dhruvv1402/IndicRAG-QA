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


@corpus_app.command("segment")
def corpus_segment() -> None:
    """Segment every document into passages with stable IDs."""
    from .corpus.segment import segment_document

    cfg = get_settings()
    docs = list(read_jsonl(cfg.manifest_path, Document))
    if not docs:
        raise typer.BadParameter(f"no manifest at {cfg.manifest_path}; run `corpus fetch` first")

    passages: list[Passage] = []
    for doc in docs:
        path = cfg.text_dir / f"{doc.doc_id}.txt"
        if not path.exists():
            continue
        got = segment_document(doc, path.read_text(encoding="utf-8"))
        passages.extend(got)
        typer.echo(f"  {doc.doc_id:28s} {len(got):4d} passages")

    n = write_jsonl(cfg.passages_path, passages)
    typer.echo("")
    typer.echo(f"{n} passages -> {cfg.passages_path}")


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
        "evals/answerable-candidates.jsonl,evals/hindi-candidates.jsonl,evals/hindi-candidates-2.jsonl",
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
) -> None:
    """Interactive review. Saves after every decision; resumable."""
    from .dataset.verify import verify_loop

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not path.exists():
        raise typer.BadParameter(f"no candidates at {path}; run `dataset generate` first")

    progress = verify_loop(
        path,
        passages,
        annotator=annotator,
        ask=typer.prompt,
        say=typer.echo,
        limit=limit or None,
    )
    typer.echo("")
    typer.echo(str(progress))


@dataset_app.command("stats")
def dataset_stats(
    path: Path = typer.Option(GOLD_PATH, "--path"),
    report: Path = typer.Option(None, "--report"),
) -> None:
    """Coverage against the PRD matrix. Run this during annotation, not after."""
    from .dataset.split import coverage, format_coverage
    from .dataset.verify import progress_of

    items = list(read_jsonl(path, QAItem))
    if not items:
        raise typer.BadParameter(f"no items at {path}")
    lines = format_coverage(coverage(items)) + ["", str(progress_of(items))]
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


@eval_app.command("qa")
def eval_qa(
    gold: Path = typer.Option(GOLD_PATH, "--gold"),
    report: Path = typer.Option(None, "--report"),
    gguf: str = typer.Option("", "--gguf", help="GGUF model; omit for the extractive baseline."),
    arms: str = typer.Option("A,B,C,D", "--arms"),
    k: int = typer.Option(5, "--k"),
    no_model: bool = typer.Option(False, "--no-model", help="Extractive provider only."),
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
    items = [i for i in read_jsonl(gold, QAItem) if i.answerable]
    if not items:
        raise typer.BadParameter(f"no answerable items in {gold}")

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

    result = run_module4(
        items, passages, complete,
        retrievers=retrievers,
        cache=GenerationCache(cfg.gen_dir / f"{model}.jsonl"),
        model=model, k=k,
        arms=[a.strip().upper() for a in arms.split(",") if a.strip()],
        progress=typer.echo,
    )
    _emit(format_module4(result), report)


@app.command("ask")
def ask(
    query: str = typer.Argument(..., help="The question, in English, Hindi or Hinglish."),
    k: int = typer.Option(5, "--k"),
    method: str = typer.Option("bm25", "--method", help="bm25 | tfidf | dense | hybrid"),
) -> None:
    """Answer one question and print the full response object."""
    from .pipeline import answer_query

    cfg = get_settings()
    passages = list(read_jsonl(cfg.passages_path, Passage))
    if not passages:
        raise typer.BadParameter("no passages; run `corpus segment` first")

    result = answer_query(query, passages, method=method, k=k)
    import json

    typer.echo(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
