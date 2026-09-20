"""The encoder registry.

Five encoders, chosen to make a specific comparison rather than to maximise one
number. Two of them are expected to lose, and that is the point.

MuRIL and IndicBERT are **masked-language-model checkpoints, not sentence
encoders**. They were pretrained with an MLM objective and have no pooling layer
trained for sentence representation, so mean-pooling their token embeddings gives
vectors whose similarity structure is dominated by lexical and positional overlap
rather than meaning -- the anisotropy problem well documented for raw BERT-family
embeddings. They are included because the brief names them, because practitioners
reach for them by name, and because the contrast makes the point that *retrieval
training matters more than pretraining-language coverage* (hypothesis H6).

To keep that result honest rather than a pooling artefact, both MLM checkpoints
are evaluated with mean pooling **and** CLS pooling, and the better of the two is
reported. Anything less and the finding could be dismissed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderSpec:
    """One encoder and everything needed to use it correctly.

    `query_prefix` / `passage_prefix` exist because E5 models are trained with
    asymmetric prefixes and measurably degrade without them. Applying them inside
    the encoding layer rather than leaving it to the caller means there is no way
    to forget, and no way for the index and the query path to disagree.
    """

    name: str
    params_m: int
    kind: str  # "sentence" | "mlm"
    role: str
    query_prefix: str = ""
    passage_prefix: str = ""
    pooling: str = "mean"
    available: bool = True
    note: str = ""

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")

    @property
    def is_mlm(self) -> bool:
        return self.kind == "mlm"


REGISTRY: dict[str, EncoderSpec] = {
    spec.name: spec
    for spec in [
        EncoderSpec(
            name="intfloat/multilingual-e5-base",
            params_m=278,
            kind="sentence",
            role="primary",
            query_prefix="query: ",
            passage_prefix="passage: ",
            note="Trained with a retrieval objective; the asymmetric prefixes are required.",
        ),
        EncoderSpec(
            name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            params_m=118,
            kind="sentence",
            role="speed baseline",
            note="Fastest CPU option; establishes what a small model gives up.",
        ),
        EncoderSpec(
            name="sentence-transformers/LaBSE",
            params_m=471,
            kind="sentence",
            role="cross-lingual specialist",
            note="Trained for bitext mining across 109 languages; expected to lead EN<->HI.",
        ),
        EncoderSpec(
            name="google/muril-base-cased",
            params_m=236,
            kind="mlm",
            role="Indic-pretrained (MLM)",
            note="Named in the brief. Not a sentence encoder -- see module docstring.",
        ),
        EncoderSpec(
            name="ai4bharat/indic-bert",
            params_m=33,
            kind="mlm",
            role="Indic-pretrained, small (MLM)",
            available=False,
            note=(
                "Named in the brief, but the repository is gated: downloading it returns "
                "401 without an accepted licence and an HF token. Excluded rather than "
                "worked around, and recorded as excluded so the omission is visible. "
                "MuRIL carries the Indic-pretrained-MLM arm on its own."
            ),
        ),
    ]
}

#: Evaluation order: cheapest first, so a partial run still yields a comparison.
#: Gated or otherwise unavailable models are filtered out rather than removed from
#: the registry, so that `index stats` can still report why they are missing.
DEFAULT_ORDER = [
    name
    for name in [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/multilingual-e5-base",
        "sentence-transformers/LaBSE",
        "google/muril-base-cased",
        "ai4bharat/indic-bert",
    ]
    if REGISTRY[name].available
]


def get(name: str) -> EncoderSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; known: {', '.join(REGISTRY)}")
    return REGISTRY[name]
