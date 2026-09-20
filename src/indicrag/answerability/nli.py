"""Natural-language-inference check: does the cited passage entail the answer?

**This is the signal the measured Module 5 result says is necessary.** A
retrieval-score threshold caught 0.000 of false-premise questions and 0.143 of
near-misses, and that is a property of the signal rather than of its calibration.
A retrieval score measures whether a question is *about* something in the corpus.
It cannot measure whether the specific asserted fact *exists*, because a question
presupposing a benefit a scheme does not provide still matches that scheme's
passages on topic, name and vocabulary. No threshold recovers information the
feature never carried.

Entailment can see it. Given the retrieved passage as premise and the generated
answer as hypothesis, a fabricated answer is not entailed however well the
question matched the topic. The same score does double duty as the brief's
"citation verification" bonus item: an answer its own cited passage does not
entail is, by definition, an unsupported citation.

`mDeBERTa-v3-base-xnli` is used because it is genuinely multilingual. That is not
a convenience here but a requirement: in the cross-lingual cells the premise is
English and the hypothesis Hindi, or the reverse, and a monolingual entailment
model would fail exactly on the slice this project exists to measure.

The model is loaded lazily and the scorer reports `available()` honestly, so the
evaluation harness degrades to the retrieval-only signals rather than failing
when the weights are absent -- which is the normal state on a machine that has
not downloaded them yet.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

#: XNLI label order for this checkpoint. Verified from the model config at load
#: time rather than assumed -- a transposed label map would invert every verdict
#: and still produce plausible-looking numbers, which is the kind of error that
#: survives review.
_EXPECTED_LABELS = ("entailment", "neutral", "contradiction")


@dataclass
class NLIVerdict:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        best = max(
            ("entailment", self.entailment),
            ("neutral", self.neutral),
            ("contradiction", self.contradiction),
            key=lambda kv: kv[1],
        )
        return best[0]

    def as_dict(self) -> dict:
        return {
            "entailment": round(self.entailment, 4),
            "neutral": round(self.neutral, 4),
            "contradiction": round(self.contradiction, 4),
        }


def _key(premise: str, hypothesis: str) -> str:
    h = hashlib.sha256()
    h.update(premise.encode("utf-8"))
    h.update(b"\x00")
    h.update(hypothesis.encode("utf-8"))
    return h.hexdigest()[:16]


class NLIScorer:
    """Entailment scoring with a disk cache.

    Roughly 0.3 s per pair on this CPU, so a full sweep over four arms is tens of
    minutes. The cache makes re-running the analysis free, in keeping with the
    rest of the evaluation: nothing expensive is computed twice.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        cache_path: Path | None = None,
        threads: int = 4,
        max_length: int = 384,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.threads = threads
        self._model = None
        self._tok = None
        self._label_index: dict[str, int] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, NLIVerdict] = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._cache[rec["key"]] = NLIVerdict(
                        rec["entailment"], rec["neutral"], rec["contradiction"]
                    )

    # --- availability -----------------------------------------------------------

    @staticmethod
    def available(model_id: str = MODEL_ID) -> bool:
        """True when the weights are present locally, without downloading them.

        Checked by loading the config rather than by calling `snapshot_download`
        with `local_files_only`. The latter reported False for a model that then
        loaded and ran perfectly: the weights were fetched with `allow_patterns`
        to skip the duplicate `.bin` checkpoint, so the snapshot is legitimately
        partial and `snapshot_download` treats any absent file as a cache miss.
        A config that loads offline is the honest test of "can this model run
        here", since `from_pretrained` is what actually has to succeed.
        """
        try:
            from transformers import AutoConfig

            AutoConfig.from_pretrained(model_id, local_files_only=True)
            return True
        except Exception:
            return False

    # --- model ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(self.threads)
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval()

        # Read the label order from the config rather than assuming it. A
        # transposed map inverts every verdict while still producing numbers
        # that look reasonable.
        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        self._label_index = {v: k for k, v in id2label.items()}
        missing = [lab for lab in _EXPECTED_LABELS if lab not in self._label_index]
        if missing:
            raise RuntimeError(
                f"{self.model_id} does not expose the expected NLI labels "
                f"(missing {missing}; got {sorted(self._label_index)})"
            )

    # --- scoring ----------------------------------------------------------------

    def score_pairs(
        self, pairs: Sequence[tuple[str, str]], *, batch_size: int = 8
    ) -> list[NLIVerdict]:
        """Score (premise, hypothesis) pairs, using the cache where possible."""
        results: list[NLIVerdict | None] = [None] * len(pairs)
        todo: list[int] = []
        for i, (premise, hypothesis) in enumerate(pairs):
            if not premise.strip() or not hypothesis.strip():
                results[i] = NLIVerdict(0.0, 1.0, 0.0)
                continue
            hit = self._cache.get(_key(premise, hypothesis))
            if hit is not None:
                results[i] = hit
            else:
                todo.append(i)

        if todo:
            self._load()
            torch = self._torch
            ent = self._label_index["entailment"]
            neu = self._label_index["neutral"]
            con = self._label_index["contradiction"]

            for start in range(0, len(todo), batch_size):
                chunk = todo[start : start + batch_size]
                premises = [pairs[i][0] for i in chunk]
                hypotheses = [pairs[i][1] for i in chunk]
                enc = self._tok(
                    premises,
                    hypotheses,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    probs = self._model(**enc).logits.softmax(dim=-1)
                for row, i in zip(probs, chunk, strict=True):
                    verdict = NLIVerdict(
                        float(row[ent]), float(row[neu]), float(row[con])
                    )
                    results[i] = verdict
                    self._remember(pairs[i][0], pairs[i][1], verdict)

        return [r if r is not None else NLIVerdict(0.0, 1.0, 0.0) for r in results]

    def entails(self, premise: str, hypothesis: str) -> float:
        return self.score_pairs([(premise, hypothesis)])[0].entailment

    def _remember(self, premise: str, hypothesis: str, verdict: NLIVerdict) -> None:
        key = _key(premise, hypothesis)
        self._cache[key] = verdict
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"key": key, **verdict.as_dict()}) + "\n")
