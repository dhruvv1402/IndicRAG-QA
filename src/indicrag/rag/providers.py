"""Answer generation behind one interface.

Four backends: extractive, local GGUF (llama.cpp), a Transformers fallback, and
any OpenAI-compatible HTTP API -- which covers Groq and Gemini. The generative
ones answer through `grounded_answer`, so each is asked the same question with
the same prompt and judged by the same parser and refusal rules.

The original three: The extractive one is not a placeholder -- it is a genuine
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
    """CPU generation through `transformers`. The fallback, not the default.

    `LlamaCppProvider` is the primary backend: a Q4_K_M GGUF needs 1.0 GB resident
    against this provider's 5.7 GB at float32, and runs roughly four times faster.
    This exists as the no-GGUF path and because it needs nothing beyond the
    encoder stack already installed for MuRIL.

    A note on how the llama.cpp dependency was resolved, because the detour cost
    real time. `llama-cpp-python` has no wheel on PyPI for this Python/platform
    and falls back to compiling, and three build attempts failed -- MSVC's
    environment uninitialised, then CMake resolving to MinGW's CMake 4.0 against
    an MSVC/Ninja toolchain. The project publishes prebuilt CPU wheels at
    `https://abetlen.github.io/llama-cpp-python/whl/cpu`, and installing from
    there took 66 milliseconds. Try the project's own wheel index before
    concluding a package must be built from source.

    Like the llama.cpp provider, this one extracts the first balanced JSON object
    from whatever the model emits, retries once with a stricter instruction, and
    counts every failure. Unparseable generations are not a random sample -- they
    skew towards longer, messier passages -- so `parse_failure_rate` is reported
    with the dataset rather than left to be discovered later.
    """

    name = "transformers"

    #: Measured on the development machine (i5-11320H, 4 threads, AVX-512 without
    #: AVX512-BF16) for Qwen2.5-1.5B-Instruct geometry. These drive `choose_dtype`.
    #:
    #: dtype     weights   decode     prefill
    #: float32    5.7 GB   4.2 tok/s   69 tok/s
    #: bfloat16   2.9 GB   6.2 tok/s   26 tok/s
    #:
    #: The split is not a quirk, it is the two regimes. Decode is batch-1 and
    #: memory-bandwidth-bound, so halving the bytes per weight makes bfloat16
    #: ~50% faster. Prefill is a large GEMM and compute-bound, and this CPU has
    #: AVX-512 but *not* AVX512-BF16, so bfloat16 matmuls are emulated and run
    #: ~2.7x slower. Prompts here carry a whole passage, so prefill dominates the
    #: total and float32 wins overall -- provided it fits.
    WEIGHTS_GB = {"float32": 5.7, "bfloat16": 2.9}
    HEADROOM_GB = 1.8  # activations, KV cache, tokenizer, interpreter

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        max_new_tokens: int = 160,
        seed: int = 20260922,
        threads: int = 4,
        dtype: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self._torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.dtype_name = self.choose_dtype() if dtype == "auto" else dtype

        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, self.dtype_name)
        )
        self._model.eval()
        self.calls = 0
        self.parse_failures = 0
        self.retries = 0

    @classmethod
    def free_ram_gb(cls) -> float | None:
        """Available physical memory, or None if it cannot be determined."""
        try:
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / 2**30
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 2**20
        except Exception:
            pass
        return None

    @classmethod
    def choose_dtype(cls) -> str:
        """float32 when it fits, bfloat16 when it does not.

        float32 is the faster choice overall here because prefill dominates, but
        it needs 5.7 GB resident against roughly 7.9 GB free on this machine --
        enough, with little to spare. If it does not fit, the failure mode is not
        a graceful slowdown: the model spills to the page file and throughput
        collapses by one to two orders of magnitude, which is far worse than
        bfloat16's ~40% penalty. So the fit is checked rather than assumed.
        """
        free = cls.free_ram_gb()
        if free is None:
            return "bfloat16"  # unknown: take the option that cannot thrash
        if free >= cls.WEIGHTS_GB["float32"] + cls.HEADROOM_GB:
            return "float32"
        return "bfloat16"

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

    #: JSON Schema for a QA candidate. Passed through `response_format`, which
    #: llama.cpp compiles into a GBNF grammar internally, so the requested object
    #: shape is the only reachable output and `kind` cannot be an invented value.
    #:
    #: The explicit `LlamaGrammar.from_string` path is NOT used, despite being the
    #: documented one: on llama-cpp-python 0.3.35 it crashes the interpreter with
    #: `OSError: access violation reading 0x0` inside `llama_sampler_sample`.
    #: Plain generation and schema-constrained generation both work on the same
    #: build, so the bug is in the grammar sampler wiring rather than in the
    #: model or the install. `response_format` gets the same guarantee by a route
    #: that does not segfault.
    #: Opt-in only, via `complete(..., schema=QA_SCHEMA)`, and not used by
    #: default: schema-constrained decoding was measured slower and no more
    #: reliable than plain generation here.
    #:
    #:   no schema            2.6s   0.05 s/token
    #:   schema               15.2s  0.19 s/token
    #:   schema + maxLength   7.0s   0.13 s/token
    #:
    #: Grammar checking against a 151k vocabulary costs more per token than the
    #: forward pass. And it did not buy reliability: on Hindi passages the schema
    #: path parsed 60% against 64% unconstrained, because `maxLength` did not
    #: stop the model running past the token cap. The real Hindi failure was
    #: elsewhere entirely -- see `dataset/prompts.py` on translated JSON keys.
    QA_SCHEMA: dict = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "maxLength": 180},
            "answer": {"type": "string", "maxLength": 220},
            "kind": {
                "type": "string",
                "enum": ["fact", "number", "date", "eligibility", "process"],
            },
        },
        "required": ["question", "answer", "kind"],
    }

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
        self.model_path = gguf_path
        self.calls = 0
        self.parse_failures = 0
        self.retries = 0

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.calls if self.calls else 0.0

    def complete(
        self,
        prompt: str,
        grammar: str | None = None,
        *,
        schema: dict | None = None,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        """Generate one JSON object.

        `grammar` is accepted for interface compatibility with the GBNF text the
        callers hold; `schema` is the route that actually constrains decoding,
        and it defaults to None because unconstrained generation measured both
        faster and marginally more reliable (see QA_SCHEMA above).

        Defaulting it to None is not merely a speed choice. `QA_SCHEMA` used to
        be applied unconditionally here, which silently broke every caller whose
        output shape is not a QA candidate: the Module 4 answering prompt asks
        for `{answerable, answer, citation, confidence}`, and a grammar pinned to
        `{question, answer, kind}` made three of those four fields *unreachable*.
        The generations still parsed, so nothing raised -- `citation` simply came
        back empty for every arm and `answerable` defaulted to true, which would
        have reported a 0.0 Citation Support Rate as a finding about the model
        rather than a bug in the harness. A constraint that silently rewrites the
        output contract of its callers has to be opt-in.
        """
        self.calls += 1
        response_format = (
            {"type": "json_object", "schema": schema} if schema is not None else None
        )
        try:
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = out["choices"][0]["message"]["content"] or ""
        except Exception:
            self.parse_failures += 1
            return ""
        found = extract_json_object(text)
        if found is None:
            self.parse_failures += 1
            return ""
        return found

    def answer(
        self, query: str, passages: Sequence[Passage], *, lang: LangIdResult
    ) -> Generated:
        return grounded_answer(self, query, passages, lang=lang)


def grounded_answer(provider, query: str, passages: Sequence[Passage], *, lang: LangIdResult) -> Generated:
    """Grounded answering for the interactive path, shared by every generator.

    Deliberately built on the same prompt and parser as the Module 4 arms
    rather than on its own. If the demo answered by a different route from
    the evaluation, the thing being demonstrated would not be the thing
    being measured, and any divergence between them would surface as a
    confusing bug report rather than as a number.

    A reply the model declines, or one that will not parse, becomes a
    refusal carrying the required string from PRD §9 -- not an empty answer
    presented as though the system had responded.
    """
    from .arms import parse_answer
    from .prompts import REFUSAL, build_answer_prompt

    data = parse_answer(provider.complete(build_answer_prompt(query, passages)))
    by_id = {p.passage_id: p for p in passages}

    if data is None:
        return Generated(
            text=REFUSAL,
            lang=lang.lang,
            confidence=0.0,
            answerable=False,
            explanation="The generator produced no parseable JSON object.",
            citations=[],
        )

    text = (data.get("answer") or "").strip()
    answerable = bool(data.get("answerable", True)) and bool(text)
    cited = (data.get("citation") or "").strip()
    # Fall back to the top passage when the model names an id that is not in
    # the context. Its own citation is preferred, but an invented one must
    # not become the displayed evidence.
    citations = [cited] if cited in by_id else ([passages[0].passage_id] if passages else [])

    if not answerable:
        return Generated(
            text=REFUSAL,
            lang=lang.lang,
            confidence=float(data.get("confidence") or 0.0),
            answerable=False,
            explanation="The generator reported the passages do not contain the answer.",
            citations=citations,
        )

    source = by_id.get(citations[0]) if citations else None
    where = f" ({source.scheme}, {source.section_path or 'lead'})" if source else ""
    return Generated(
        text=text,
        lang=source.lang if source else lang.lang,
        confidence=float(data.get("confidence") or 0.0),
        answerable=True,
        explanation=(
            f"Generated from {citations[0]}{where}"
            + ("" if cited in by_id else ", which the model did not itself cite")
            + "."
        ),
        citations=citations,
    )


class APIError(RuntimeError):
    """An API call that failed for a reason worth showing the user."""


#: Hosted endpoints that speak the OpenAI chat-completions protocol. The model
#: is only a default; model names change, so `--model` overrides it.
API_PRESETS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
}


class OpenAICompatProvider:
    """Any OpenAI-compatible chat-completions API: Groq, Gemini, a local server.

    Standard library only -- no SDK dependency for a single POST. Temperature 0
    matches the local generator, so a hosted model and the 3B model differ in
    the model, not in the sampling.

    **What leaves the machine.** Every call sends the question and the retrieved
    passages to the provider. The passages are public Wikipedia text; the
    question is whatever the user typed. The key is read from the environment
    and never logged or included in an error message.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        label: str = "api",
        max_tokens: int = 320,
        timeout: float = 60.0,
        retries: int = 3,
        opener=None,
        sleep=None,
    ):
        import time
        import urllib.request

        if not api_key:
            raise APIError(f"no API key for {label}; set it in the environment or in .env")
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.model = model
        self.name = f"{label}:{model}"
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.parse_failures = 0
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep or time.sleep

    @classmethod
    def from_preset(cls, preset: str, *, model: str = "", api_key: str = "", **kw) -> OpenAICompatProvider:
        import os

        if preset not in API_PRESETS:
            raise APIError(f"unknown API {preset!r}; choose one of {', '.join(API_PRESETS)}")
        cfg = API_PRESETS[preset]
        key = api_key or os.environ.get(cfg["key_env"], "") or os.environ.get("INDICRAG_LLM_API_KEY", "")
        if not key:
            raise APIError(f"{cfg['key_env']} is not set; add it to your environment or to .env")
        return cls(base_url=cfg["base_url"], api_key=key, model=model or cfg["model"], label=preset, **kw)

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.calls if self.calls else 0.0

    def _post(self, payload: dict) -> dict:
        import json
        import urllib.error
        import urllib.request

        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body, method="POST",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
            )
            try:
                with self._open(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    err = json.loads(exc.read().decode("utf-8"))
                    err = err[0] if isinstance(err, list) and err else err
                    inner = err.get("error", err) if isinstance(err, dict) else err
                    detail = inner.get("message", "") if isinstance(inner, dict) else str(inner)
                except Exception:  # noqa: BLE001 -- the status code is still reported
                    pass
                # Rate limits and transient server errors are retried with backoff,
                # honouring Retry-After when the provider sends one.
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    wait = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = float(wait) if wait else 2.0 ** attempt
                    except ValueError:
                        delay = 2.0 ** attempt
                    self._sleep(min(delay, 20.0))
                    continue
                raise APIError(f"{self.name} returned HTTP {exc.code}: {detail or exc.reason}") from None
            except urllib.error.URLError as exc:
                if attempt < self.retries:
                    self._sleep(2.0 ** attempt)
                    continue
                raise APIError(f"could not reach {self.base_url}: {exc.reason}") from None
        raise APIError(f"{self.name}: retries exhausted")

    def complete(self, prompt: str, grammar: str | None = None, **_: object) -> str:
        """One chat completion, reduced to the JSON object it contains.

        `grammar` is accepted for interface compatibility and ignored: hosted
        APIs do not take GBNF, and the answer prompt already asks for JSON,
        which `extract_json_object` recovers from any surrounding prose.
        """
        self.calls += 1
        out = self._post({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        })
        try:
            text = out["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            self.parse_failures += 1
            return ""
        found = extract_json_object(text)
        if found is None:
            self.parse_failures += 1
            return ""
        return found

    def answer(
        self, query: str, passages: Sequence[Passage], *, lang: LangIdResult
    ) -> Generated:
        return grounded_answer(self, query, passages, lang=lang)
