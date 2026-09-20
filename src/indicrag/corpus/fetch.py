"""Fetch the bilingual corpus and record its provenance.

The Hindi counterpart of each scheme is resolved through the MediaWiki
`langlinks` API rather than being typed into the registry, so a pair is used only
where Wikipedia itself asserts the two articles describe the same subject. That
is a stronger guarantee of parallelism than matching titles by hand, and it is
what makes cross-lingual retrieval a real measurement here rather than an
assumption.

Every fetch is recorded in the manifest with a `sha256` and a `retrieved_at`
timestamp. Wikipedia articles change daily, so without those the paper cannot
state which snapshot produced its numbers.

The retry logic is not defensive padding. A plain loop over ~50 API calls reliably
hits a non-JSON error body partway through -- observed during development as a
`JSONDecodeError` on call 4 of the extract pass -- because the API rate-limits and
returns an HTML error page rather than JSON. Backoff plus a JSON-shape check is
the difference between a corpus build that completes and one that dies halfway
with half a manifest written.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from ..models import Document
from .sources import ATTRIBUTION, LICENCE, SchemeSource

USER_AGENT = "IndicRAG-QA/0.1 (CSET346 academic project; https://github.com/dhruvv1402/IndicRAG-QA)"

EN_API = "https://en.wikipedia.org/w/api.php"
HI_API = "https://hi.wikipedia.org/w/api.php"

#: Wikipedia asks for serial requests from unregistered clients. This is polite
#: rather than required, and the whole corpus is ~50 calls, so it costs seconds.
REQUEST_DELAY_S = 0.4
MAX_RETRIES = 4


class FetchError(RuntimeError):
    pass


@dataclass
class FetchResult:
    documents: list[Document]
    pairs: list[tuple[str, str]]
    missing: list[str]
    unpaired: list[str]

    def summary(self) -> str:
        return (
            f"{len(self.documents)} documents from {len(self.pairs)} EN/HI pairs; "
            f"{len(self.missing)} missing, {len(self.unpaired)} without a Hindi article"
        )


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _api(session: requests.Session, url: str, **params: Any) -> dict:
    """Call the MediaWiki API with backoff, verifying the body is actually JSON.

    `response.json()` raising is the observed failure mode under rate limiting --
    the API answers 200 with an HTML error page -- so a bare `raise_for_status()`
    is not enough to catch it.
    """
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                raise FetchError("rate limited")
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise FetchError(str(data["error"]))
            return data
        except Exception as exc:  # noqa: BLE001 -- retried below, re-raised at the end
            last = exc
            time.sleep(2**attempt * 0.75)
    raise FetchError(f"{url} failed after {MAX_RETRIES} attempts: {last}")


def resolve_pairs(
    schemes: Iterable[SchemeSource], session: requests.Session | None = None
) -> tuple[list[tuple[SchemeSource, str, str]], list[str], list[str]]:
    """Resolve each English title to its Hindi counterpart via `langlinks`.

    Returns (resolved, missing, unpaired). `missing` is an English title Wikipedia
    does not have; `unpaired` is one it has but with no Hindi article. Both are
    reported rather than silently dropped -- the corpus manifest should show what
    was attempted, not only what succeeded.
    """
    session = session or _session()
    schemes = list(schemes)
    by_title = {s.en_title: s for s in schemes}
    resolved: list[tuple[SchemeSource, str, str]] = []
    missing: list[str] = []
    unpaired: list[str] = []

    for i in range(0, len(schemes), 10):
        batch = schemes[i : i + 10]
        data = _api(
            session,
            EN_API,
            action="query",
            titles="|".join(s.en_title for s in batch),
            prop="langlinks",
            lllang="hi",
            lllimit="500",
            redirects="1",
        )
        query = data.get("query", {})
        # A redirect means the canonical title differs from what we asked for;
        # map it back so the scheme slug still attaches to the right page.
        redirects = {r["to"]: r["from"] for r in query.get("redirects", [])}
        for page in query.get("pages", []):
            title = page.get("title", "")
            origin = redirects.get(title, title)
            scheme = by_title.get(origin) or by_title.get(title)
            if page.get("missing"):
                missing.append(origin)
                continue
            links = page.get("langlinks") or []
            if not links:
                unpaired.append(title)
                continue
            if scheme is None:
                continue
            resolved.append((scheme, title, links[0]["title"]))
        time.sleep(REQUEST_DELAY_S)

    return resolved, missing, unpaired


def _extract_plaintext(session: requests.Session, api_url: str, title: str) -> tuple[str, str]:
    """Return (plaintext, canonical title) for one article.

    `explaintext` keeps the `== Section ==` markers, which is the whole reason
    this is preferred over the HTML endpoint: they become `section_path` metadata
    during segmentation, and a passage that knows which clause it came from is
    far more useful in an error case than a bare offset.
    """
    data = _api(
        session,
        api_url,
        action="query",
        titles=title,
        prop="extracts",
        explaintext="1",
        exlimit="1",
        redirects="1",
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise FetchError(f"no extract for {title}")
    return pages[0].get("extract", "") or "", pages[0].get("title", title)


def fetch_corpus(
    schemes: Iterable[SchemeSource],
    text_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> FetchResult:
    """Fetch every resolvable EN/HI pair, write the text, return the manifest rows."""
    session = _session()
    text_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda _m: None)

    resolved, missing, unpaired = resolve_pairs(schemes, session)
    say(f"resolved {len(resolved)} pairs ({len(missing)} missing, {len(unpaired)} unpaired)")

    documents: list[Document] = []
    pairs: list[tuple[str, str]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for scheme, en_title, hi_title in resolved:
        row: list[Document] = []
        for lang, api_url, title in (("en", EN_API, en_title), ("hi", HI_API, hi_title)):
            try:
                text, canonical = _extract_plaintext(session, api_url, title)
            except FetchError as exc:
                say(f"  !! {scheme.slug}-{lang}: {exc}")
                continue
            if len(text) < 500:
                say(f"  -- {scheme.slug}-{lang}: only {len(text)} chars, skipped")
                continue
            doc_id = f"{scheme.slug}-{lang}"
            path = text_dir / f"{doc_id}.txt"
            path.write_text(text, encoding="utf-8", newline="\n")
            host = "en" if lang == "en" else "hi"
            row.append(
                Document(
                    doc_id=doc_id,
                    scheme=scheme.slug,
                    lang=lang,
                    title=canonical,
                    ministry=scheme.category,
                    source_url=f"https://{host}.wikipedia.org/wiki/{canonical.replace(' ', '_')}",
                    retrieved_at=now,
                    sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    licence=LICENCE,
                    format="wikitext-plain",
                    pages=0,
                    ocr_used=False,
                    notes=ATTRIBUTION,
                )
            )
            time.sleep(REQUEST_DELAY_S)

        # Only keep a scheme when BOTH languages landed. A one-sided scheme would
        # silently bias the language-pair matrix in docs/PRD.md §6.2, because
        # questions could never be written against the missing side.
        if len(row) == 2:
            documents.extend(row)
            pairs.append((scheme.slug, hi_title))
            say(f"  ok {scheme.slug}: en={row[0].sha256[:7]} hi={row[1].sha256[:7]}")
        else:
            say(f"  -- {scheme.slug}: incomplete pair, dropped")
            unpaired.append(scheme.en_title)

    return FetchResult(documents=documents, pairs=pairs, missing=missing, unpaired=unpaired)
