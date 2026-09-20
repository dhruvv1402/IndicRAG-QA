"""Answer generation behind one interface.

Three backends. The extractive one is not a placeholder -- it is a genuine
non-neural QA baseline and the `--no-model` fast path, and having it means the
retrieval and answerability halves of the project are testable, evaluable and
demonstrable without a 2 GB model download. Every result it produces is by
construction a span of a retrieved passage, so its Citation Support Rate is 1.0
by definition, which makes it a useful floor when reading the generative arms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..corpus.normalize import split_sentences
from ..index.tokenize import tokenize
from ..models import Passage
from ..query.langid import LangIdResult


@dataclass
class Generated:
    text: str
    lang: str = ""
    confidence: float = 0.0
    explanation: str = ""
    answerable: bool = True
    citations: list[str] | None = None


class Provider(Protocol):
    name: str

    def answer(
        self, query: str, passages: Sequence[Passage], *, lang: LangIdResult
    ) -> Generated: ...


class ExtractiveProvider:
    """Select the best-supporting sentence from the top passages.

    Scoring is token overlap with the query, weighted by inverse sentence length
    so a long sentence cannot win merely by containing more words. A small bonus
    is given for sentences carrying a digit when the query asks a quantity
    question ("how much", "kitna", "कितनी"), because in this corpus the answer to
    such a question is almost always the sentence with the number in it, and
    overlap alone routinely picks the topic sentence next to it instead.
    """

    name = "extractive"

    _QUANTITY_MARKERS = {
        "how", "much", "many", "amount", "rate", "limit", "age", "income",
        "kitna", "kitni", "kitne", "kab", "year", "years",
        "कितना", "कितनी", "कितने", "राशि", "आयु", "आय", "सीमा", "वर्ष",
    }

    def answer(
        self, query: str, passages: Sequence[Passage], *, lang: LangIdResult
    ) -> Generated:
        if not passages:
            return Generated(text="", confidence=0.0, answerable=False)

        q_tokens = set(tokenize(query))
        wants_number = bool(q_tokens & {t.lower() for t in self._QUANTITY_MARKERS}) or any(
            m in query for m in ("कितन", "kitn", "how much", "how many")
        )

        best: tuple[float, str, Passage] | None = None
        for rank, p in enumerate(passages):
            # Passages further down the ranking start from a lower base, so a
            # marginally better sentence in a much worse passage does not win.
            rank_decay = 1.0 / (1.0 + 0.35 * rank)
            for sent in split_sentences(p.text):
                s_tokens = tokenize(sent)
                if not s_tokens:
                    continue
                overlap = len(q_tokens & set(s_tokens))
                if not overlap:
                    continue
                score = overlap / (1.0 + 0.02 * len(s_tokens))
                if wants_number and any(c.isdigit() for c in sent):
                    score *= 1.6
                score *= rank_decay
                if best is None or score > best[0]:
                    best = (score, sent.strip(), p)

        if best is None:
            top = passages[0]
            sents = split_sentences(top.text)
            return Generated(
                text=sents[0] if sents else top.text[:300],
                lang=top.lang,
                confidence=0.15,
                explanation="No query term matched; returning the lead sentence of the top passage.",
                citations=[top.passage_id],
            )

        score, sentence, passage = best
        confidence = min(0.95, 0.35 + 0.12 * score)
        return Generated(
            text=sentence,
            lang=passage.lang,
            confidence=confidence,
            explanation=(
                f"Extracted verbatim from {passage.passage_id} "
                f"({passage.scheme}, {passage.section_path or 'lead'})."
            ),
            citations=[passage.passage_id],
        )


class TransformersProvider:
    """CPU generation through `transformers`, with no compiler dependency.

    **Why this exists instead of LlamaCppProvider.** `llama-cpp-python` has no
    prebuilt wheel for this Python/platform and must compile. Three build attempts
    failed on this machine: first because MSVC's environment was not initialised,
    then -- once it was -- because CMake resolves to MinGW's CMake 4.0 while the
    toolchain is MSVC/Ninja, and the compiler ABI check fails on that mismatch.
    Rather than keep grinding on a build, generation runs through `transformers`,
    which is already installed and already used for the MuRIL encoder arm.

    **What that costs, and how it is mitigated.** llama.cpp supports GBNF
    grammars, which make malformed JSON literally unreachable;
    `docs/ARCHITECTURE.md` §11.4 calls that the highest-value detail in the
    generation stage, and this path does not have it. The concern was never
    malformed output as such -- it is that unparseable generations are *not a
    random sample*. They skew towards longer, messier passages, so silently
    dropping them biases the dataset towards easy ones.

    So instead of dropping silently: extract the first balanced JSON object from
    whatever the model emits, retry once with a stricter instruction on failure,
    and **count every failure** so `parse_failure_rate` can be reported alongside
    the dataset rather than discovered later. A measured bias is a caveat; an
    unmeasured one is a flaw.
    """

    name = "transformers"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        max_new_tokens: int = 160,
        seed: int = 20260922,
        threads: int = 4,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self._torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        self._model.eval()
        self.calls = 0
        self.parse_failures = 0
        self.retries = 0

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.calls if self.calls else 0.0

    def _generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self._tok([text], return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,  # deterministic: PRD NFR-5
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def complete(self, prompt: str, grammar: str | None = None, **_: object) -> str:
        """Generate and return a JSON object string. `grammar` is accepted and
        ignored -- the signature matches LlamaCppProvider so the two are
        interchangeable if the build is ever fixed."""
        self.calls += 1
        raw = self._generate(prompt)
        found = extract_json_object(raw)
        if found is not None:
            return found

        self.retries += 1
        stricter = prompt + '\n\nReply with ONLY a JSON object. Start your reply with { and end with }.'
        raw = self._generate(stricter)
        found = extract_json_object(raw)
        if found is None:
            self.parse_failures += 1
            return ""
        return found


def extract_json_object(text: str) -> str | None:
    """Pull the first balanced `{...}` out of a model reply.

    Small models wrap JSON in prose or markdown fences even when told not to.
    Brace-counting (respecting strings and escapes) is more reliable here than a
    regex, which trips on nested braces and on braces inside string values.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class LlamaCppProvider:
    """Local GGUF generation via llama-cpp-python.

    `complete(prompt, grammar)` is the low-level call the dataset bootstrapper
    uses. The grammar argument is the important part: a GBNF grammar makes any
    output but the requested JSON shape *unreachable*, rather than merely
    requested. A 1.5B model asked politely for JSON emits malformed output often
    enough that the repair pass becomes its own source of bias, because the
    generations that fail to parse are not a random sample -- they skew towards
    the longer, messier passages, so silently dropping them biases the dataset
    towards easy ones.

    Deterministic by default: temperature 0 and a fixed seed, so a regenerated
    dataset is identical and `docs/PRD.md` NFR-5 holds.
    """

    name = "llamacpp"

    def __init__(
        self,
        gguf_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        seed: int = 20260922,
        verbose: bool = False,
    ):
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            seed=seed,
            verbose=verbose,
        )
        self._grammars: dict[str, object] = {}
        self.model_path = gguf_path

    def _grammar(self, text: str):
        """Compile and cache. Recompiling per call costs more than generation."""
        from llama_cpp import LlamaGrammar

        if text not in self._grammars:
            self._grammars[text] = LlamaGrammar.from_string(text, verbose=False)
        return self._grammars[text]

    def complete(
        self,
        prompt: str,
        grammar: str | None = None,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {"max_tokens": max_tokens, "temperature": temperature}
        if grammar:
            kwargs["grammar"] = self._grammar(grammar)
        out = self._llm.create_chat_completion(messages=messages, **kwargs)
        return out["choices"][0]["message"]["content"] or ""

    def answer(
        self, query: str, passages: Sequence[Passage], *, lang: LangIdResult
    ) -> Generated:  # pragma: no cover -- requires a model file
        raise NotImplementedError("RAG answering is P4; see docs/PLAN.md §1.5")
