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
from .models import Document, Passage, read_jsonl, write_jsonl

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
corpus_app = typer.Typer(no_args_is_help=True, help="Fetch, extract and segment the corpus.")
index_app = typer.Typer(no_args_is_help=True, help="Build lexical and dense indices.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")


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
