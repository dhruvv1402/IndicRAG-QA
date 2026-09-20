"""The corpus source registry.

**A documented deviation from docs/PRD.md §5.2, and the reason for it.**

The PRD specifies official scheme guideline PDFs from the issuing ministries. Two
things were found when that was attempted, both of which are recorded as risks R1
and R2 in docs/PLAN.md §5 and both of which duly materialised on the first
documents tried:

1. The Hindi paths on `socialjustice.gov.in` return HTTP 200 with a "Page not
   found" body -- a soft 404. The bilingual premise does not survive first
   contact there.
2. `PMS_for_SCs_Scheme_Guidelines.pdf` (18 pages, 2.8 MB) yields **17 characters**
   of extractable text. It is a scan. So are most Hindi circulars of this kind,
   and this machine has no Tesseract installed, so OCR is not available.

A corpus of English-only text-layer PDFs would defeat the entire cross-lingual
research question, which is the thing the project exists to measure. The corpus
is therefore built from **parallel English/Hindi Wikipedia articles on Indian
government welfare and education schemes**, which are:

- genuinely parallel (linked by `langlinks`, not machine-translated by us),
- text-layer by construction, with no extraction risk,
- structured with `== Section ==` headings that give real `section_path` metadata,
- CC BY-SA licensed and attributable.

What this costs, stated plainly so it lands in the paper's limitations rather
than being discovered by a reader: the register is **encyclopedic rather than
regulatory**. Wikipedia describes schemes; it does not legislate them. Numbers
and eligibility criteria appear but are secondary prose, not normative clauses,
and they may lag the current circular. Every research question in docs/PRD.md §2
remains testable -- the corpus is still bilingual, still full of near-duplicate
scheme descriptions, still rich in amounts, dates and named entities -- but a
claim about retrieval over *statutory* text is not supported by these results.

Scheme titles are seeded by hand and the Hindi counterpart is resolved through
the MediaWiki `langlinks` API rather than typed here, so a pair is only ever used
if Wikipedia itself asserts the two articles are the same subject.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemeSource:
    """One scheme, identified by its English Wikipedia title.

    `slug` is the stable key used to build `doc_id` (`<slug>-en`, `<slug>-hi`).
    It must not change once passages are segmented: the gold set, the embedding
    caches and the error analysis all reference passage IDs derived from it.
    """

    slug: str
    en_title: str
    category: str


#: Seeded by hand, filtered by whether Wikipedia has a Hindi counterpart.
#: Skewed towards education and scholarship schemes to keep the domain coherent,
#: with welfare schemes included because they share clause structure -- income
#: ceilings, eligibility tiers, documentation lists -- which is exactly the
#: near-duplicate pressure that makes retrieval non-trivial (docs/PRD.md §5.2).
SCHEMES: list[SchemeSource] = [
    # --- Education and skills -------------------------------------------------
    SchemeSource("sarva-shiksha", "Sarva Shiksha Abhiyan", "education"),
    SchemeSource("rte-act", "Right of Children to Free and Compulsory Education Act, 2009", "education"),
    SchemeSource("midday-meal", "Midday Meal Scheme", "education"),
    SchemeSource("pmkvy", "Pradhan Mantri Kaushal Vikas Yojana", "skills"),
    SchemeSource("skill-india", "Skill India", "skills"),
    SchemeSource("ddu-gky", "Deen Dayal Upadhyaya Grameen Kaushalya Yojana", "skills"),
    SchemeSource("icds", "Integrated Child Development Services", "education"),
    # --- Financial inclusion and pensions -------------------------------------
    SchemeSource("jan-dhan", "Pradhan Mantri Jan Dhan Yojana", "financial"),
    SchemeSource("atal-pension", "Atal Pension Yojana", "pension"),
    SchemeSource("nps", "National Pension System", "pension"),
    SchemeSource("sukanya", "Sukanya Samriddhi Account", "financial"),
    SchemeSource("mudra", "Pradhan Mantri Mudra Yojana", "financial"),
    SchemeSource("stand-up-india", "Stand-Up India", "financial"),
    # --- Social welfare -------------------------------------------------------
    SchemeSource("nsap", "National Social Assistance Scheme", "welfare"),
    SchemeSource("mgnrega", "Mahatma Gandhi National Rural Employment Guarantee Act", "welfare"),
    SchemeSource("awas-yojana", "Pradhan Mantri Awas Yojana", "housing"),
    SchemeSource("garib-kalyan-anna", "Pradhan Mantri Garib Kalyan Anna Yojana", "welfare"),
    SchemeSource("ujjwala", "Pradhan Mantri Ujjwala Yojana", "welfare"),
    # --- Health and women / child ---------------------------------------------
    SchemeSource("ayushman-bharat", "Ayushman Bharat Yojana", "health"),
    SchemeSource("matru-vandana", "Pradhan Mantri Matri Vandana Yojana", "health"),
    SchemeSource("beti-bachao", "Beti Bachao Beti Padhao", "welfare"),
    # --- Agriculture and governance -------------------------------------------
    SchemeSource("pm-kisan", "Pradhan Mantri Kisan Samman Nidhi", "agriculture"),
    SchemeSource("swachh-bharat", "Swachh Bharat Mission", "sanitation"),
    SchemeSource("digital-india", "Digital India", "governance"),
    # --- Second pass ----------------------------------------------------------
    # Hindi Wikipedia articles run roughly a third the length of their English
    # counterparts, so the first 24 schemes yielded 475 passages against the
    # 800-1500 target in docs/PRD.md §5.1. These were added to close that gap.
    # Titles with no Hindi counterpart are dropped automatically by the langlinks
    # resolution in fetch.py, so an optimistic list costs nothing.
    SchemeSource("suraksha-bima", "Pradhan Mantri Suraksha Bima Yojana", "insurance"),
    SchemeSource("jeevan-jyoti", "Pradhan Mantri Jeevan Jyoti Bima Yojana", "insurance"),
    SchemeSource("janani-suraksha", "Janani Suraksha Yojana", "health"),
    SchemeSource("rsby", "Rashtriya Swasthya Bima Yojana", "health"),
    SchemeSource("nhm", "National Health Mission", "health"),
    SchemeSource("antyodaya-anna", "Antyodaya Anna Yojana", "welfare"),
    SchemeSource("pds", "Public distribution system", "welfare"),
    SchemeSource("gram-sadak", "Pradhan Mantri Gram Sadak Yojana", "infrastructure"),
    SchemeSource("jal-jeevan", "Jal Jeevan Mission", "infrastructure"),
    SchemeSource("smart-cities", "Smart Cities Mission", "infrastructure"),
    SchemeSource("amrut", "Atal Mission for Rejuvenation and Urban Transformation", "infrastructure"),
    SchemeSource("nrlm", "Deendayal Antyodaya Yojana", "livelihood"),
    SchemeSource("startup-india", "Startup India", "enterprise"),
    SchemeSource("make-in-india", "Make in India", "enterprise"),
    SchemeSource("aadhaar", "Aadhaar", "governance"),
    SchemeSource("soil-health-card", "Soil Health Card Scheme", "agriculture"),
    SchemeSource("kisan-credit-card", "Kisan Credit Card", "agriculture"),
    SchemeSource("enam", "National Agriculture Market", "agriculture"),
    SchemeSource("kgbv", "Kasturba Gandhi Balika Vidyalaya", "education"),
    SchemeSource("nlm", "National Literacy Mission Programme", "education"),
    SchemeSource("epfo", "Employees' Provident Fund Organisation", "pension"),
    SchemeSource("indira-awaas", "Indira Awaas Yojana", "housing"),
    SchemeSource("sagy", "Sansad Adarsh Gram Yojana", "rural"),
    SchemeSource("atmanirbhar", "Atmanirbhar Bharat", "enterprise"),
    SchemeSource("ayushman-digital", "Ayushman Bharat Digital Mission", "health"),
]

LICENCE = "CC BY-SA 4.0 (Wikipedia)"
ATTRIBUTION = "Wikipedia contributors, English and Hindi Wikipedia"

by_slug = {s.slug: s for s in SCHEMES}
