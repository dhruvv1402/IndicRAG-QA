"""Render paper/paper.md into an IEEE-format .docx.

Why .docx and not LaTeX. IEEEtran is the usual route and Overleaf would compile
it without a local TeX install, but this paper contains Devanagari (योग्यता),
Greek (α, κ, Δ) and typographic dashes in running text. Under pdfLaTeX those are
a compile error, and the fix -- XeLaTeX plus a Devanagari-capable font -- is
something the author would have to debug remotely, on an artefact I cannot
compile here to check. Word handles the same characters natively. Given the
choice between a format I can verify structurally and one I would be handing
over untested, the verifiable one wins.

Layout follows the IEEE conference template: US Letter, 0.75in top / 1.0in
bottom / 0.625in side margins, a single-column title block, and a two-column
body at 10pt Times New Roman with 0.2in between columns.

    python scripts/build-docx.py          # paper/IndicRAG-QA.docx
    python scripts/build-docx.py --pdf    # and paper/IndicRAG-QA.pdf, via Word

Writes paper/IndicRAG-QA.docx. Anything the converter could not handle is
listed at the end rather than dropped silently -- a converter that quietly
discards a table produces a paper missing a result.

Figures are placed where the text puts them: a line `![Fig. N. Caption](figures/x.png)`
in a section draft becomes the image at column width with its caption beneath.
The PDF is exported by Microsoft Word through COM, so it is what Word lays out,
not an approximation of it; without Word, `--pdf` says so and the .docx stands.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "paper" / "paper.md"
TARGET = ROOT / "paper" / "IndicRAG-QA.docx"
PDF = TARGET.with_suffix(".pdf")

#: One column of the two-column body: (8.5in - 2 x 0.625in - 0.2in) / 2.
COLUMN_WIDTH = Inches(3.5)
FIGURE = re.compile(r"!\[(.+?)\]\((.+?)\)")

BODY_FONT = "Times New Roman"
BODY_SIZE = Pt(10)

AUTHOR = "Dhruv"
AFFILIATION = "School of Computer Science Engineering and Technology, Bennett University"
COURSE = "CSET 346 — Natural Language Processing"


def set_columns(section, count: int, space_twips: int = 288) -> None:
    """Set the column count on a section. python-docx has no API for this."""
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(space_twips))
    cols.set(qn("w:equalWidth"), "1")


INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)", re.S)


def add_runs(paragraph, text: str) -> None:
    """Emit text as runs, honouring **bold**, *italic* and `code`."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*") and part.endswith("*"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
        else:
            run = paragraph.add_run(part)
        run.font.name = run.font.name or BODY_FONT
        if run.font.size is None:
            run.font.size = BODY_SIZE


def body_paragraph(doc, text: str, *, indent: bool = True, italic: bool = False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.first_line_indent = Inches(0.2) if indent else Inches(0)
    add_runs(p, text)
    if italic:
        for run in p.runs:
            run.italic = True
    return p


def heading(doc, text: str, *, level: int):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    if level == 1:
        # IEEE section headings are centred and set in small capitals. Word's
        # small-caps attribute leaves the leading numeral alone, which is what
        # the template wants ("I. INTRODUCTION", not "I. Introduction").
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.small_caps = True
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.italic = True
    run.font.name = BODY_FONT
    run.font.size = BODY_SIZE
    return p


def add_table(doc, rows: list[list[str]]):
    """Render a markdown table. IEEE tables are 8pt with a ruled header."""
    header, *body = rows
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.autofit = True

    for cell, text in zip(table.rows[0].cells, header, strict=False):
        cell.text = ""
        para = cell.paragraphs[0]
        para.paragraph_format.space_after = Pt(0)
        run = para.add_run(re.sub(r"[*`]", "", text))
        run.bold = True
        run.font.name = BODY_FONT
        run.font.size = Pt(8)

    for row in body:
        cells = table.add_row().cells
        for cell, text in zip(cells, row, strict=False):
            cell.text = ""
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            add_runs(para, text)
            for run in para.runs:
                run.font.size = Pt(8)
    return table


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    """Consume a markdown table starting at `start`. Returns rows and next index."""
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        raw = lines[i].strip().strip("|")
        # The |---|---| separator row carries no content.
        if not re.fullmatch(r"[\s:|-]+", raw):
            rows.append([c.strip() for c in raw.split("|")])
        i += 1
    return rows, i


def add_figure(doc, path: Path, caption: str, warnings: list[str]) -> bool:
    """Embed a figure at column width with an IEEE-style caption beneath it."""
    if not path.exists():
        warnings.append(f"figure {path.name} is missing; run scripts/build-figures.py")
        body_paragraph(doc, f"[missing figure: {path.name}]", indent=False)
        return False
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.space_before = Pt(6)
    pic.paragraph_format.space_after = Pt(2)
    pic.paragraph_format.keep_with_next = True
    pic.add_run().add_picture(str(path), width=COLUMN_WIDTH)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cap.paragraph_format.space_after = Pt(8)
    add_runs(cap, caption)
    for run in cap.runs:
        run.font.size = Pt(8)
    return True


def build() -> tuple[Document, list[str]]:
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    warnings: list[str] = []

    doc = Document()
    doc.figures = 0
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = BODY_SIZE

    first = doc.sections[0]
    first.page_width, first.page_height = Inches(8.5), Inches(11)
    first.top_margin, first.bottom_margin = Inches(0.75), Inches(1.0)
    first.left_margin, first.right_margin = Inches(0.625), Inches(0.625)
    set_columns(first, 1)

    # --- title block, single column -------------------------------------------
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Script-Aware Fusion for Cross-Lingual Retrieval:\n"
                        "Why Naive Hybrid Retrieval Fails on Indic and Code-Mixed Queries")
    run.font.size = Pt(20)
    run.font.name = BODY_FONT

    for line, size in ((AUTHOR, 11), (AFFILIATION, 10), (COURSE, 10)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(line)
        r.font.size = Pt(size)
        r.font.name = BODY_FONT
        if line == AUTHOR:
            r.font.size = Pt(11)

    # --- body, two columns ----------------------------------------------------
    body = doc.add_section(WD_SECTION.CONTINUOUS)
    body.top_margin, body.bottom_margin = Inches(0.75), Inches(1.0)
    body.left_margin, body.right_margin = Inches(0.625), Inches(0.625)
    set_columns(body, 2)

    i = 0
    in_front_matter = True
    para: list[str] = []

    def flush() -> None:
        if para:
            body_paragraph(doc, " ".join(para))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip everything before the Abstract: the build banner and the source
        # note are instructions to the author, not part of the paper.
        if in_front_matter:
            if stripped.startswith("## Abstract"):
                in_front_matter = False
                heading(doc, "Abstract", level=1)
            i += 1
            continue

        if not stripped:
            flush()
            i += 1
            continue

        if stripped.startswith("|"):
            flush()
            rows, i = parse_table(lines, i)
            if rows:
                add_table(doc, rows)
                doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue

        if re.fullmatch(r"-{3,}", stripped):
            flush()
            i += 1
            continue

        if stripped.startswith("### "):
            flush()
            heading(doc, stripped[4:], level=2)
            i += 1
            continue

        if stripped.startswith("## "):
            flush()
            heading(doc, stripped[3:], level=1)
            i += 1
            continue

        if stripped.startswith("# "):
            flush()
            i += 1
            continue

        if stripped.startswith("> "):
            flush()
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            p = body_paragraph(doc, " ".join(quote), indent=False)
            p.paragraph_format.left_indent = Inches(0.15)
            p.paragraph_format.space_after = Pt(6)
            continue

        figure = FIGURE.fullmatch(stripped)
        if figure:
            flush()
            if add_figure(doc, SOURCE.parent / figure.group(2), figure.group(1), warnings):
                doc.figures += 1
            i += 1
            continue

        if re.match(r"^(\d+\.|[-*])\s+", stripped):
            flush()
            item = re.sub(r"^(\d+\.|[-*])\s+", "", stripped)
            i += 1
            # A list item may wrap onto following indented lines.
            while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                item += " " + lines[i].strip()
                i += 1
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(0)
            add_runs(p, item)
            continue

        para.append(stripped)
        i += 1

    flush()
    return doc, warnings


def export_pdf(source: Path, target: Path) -> str | None:
    """Export through Word's own layout engine. Returns an error, or None."""
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
        f"try {{ $d = $w.Documents.Open('{source}', $false, $true); "
        f"$d.ExportAsFixedFormat('{target}', 17); $d.Close($false) }} "
        "finally { $w.Quit() }"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"could not run Word: {exc}"
    if proc.returncode != 0 or not target.exists():
        return (proc.stderr or proc.stdout or "Word export failed").strip().splitlines()[0]
    return None


def main() -> int:
    if not SOURCE.exists():
        print(f"missing {SOURCE}; run scripts/build-paper.py first")
        return 1

    doc, warnings = build()
    doc.save(TARGET)

    tables = len(doc.tables)
    paras = len(doc.paragraphs)
    print(f"wrote {TARGET.relative_to(ROOT)}  ({paras} paragraphs, {tables} tables, "
          f"{doc.figures} figures, {TARGET.stat().st_size // 1024} KB)")
    for w in warnings:
        print(f"  WARNING: {w}")

    if "--pdf" in sys.argv[1:]:
        if PDF.exists():
            PDF.unlink()
        error = export_pdf(TARGET, PDF)
        if error:
            print(f"  PDF NOT WRITTEN: {error}")
            return 1
        print(f"wrote {PDF.relative_to(ROOT)}  ({PDF.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
