"""Configuration, resolved once and cached.

Two things here are not boilerplate.

First, `load_dotenv` is called at module import in addition to pydantic-settings'
own `env_file` handling. They are not redundant: `Settings` populates *this*
object, but `llama-cpp-python`, `transformers` and `huggingface_hub` read
`os.environ` directly and never see it. Without the explicit load, `HF_HOME` set
in `.env` would be honoured by our code and ignored by the library that actually
downloads the 2 GB file.

Second, the defaults for `n_threads` and the cache directories assume the machine
described in docs/PRD.md §8: four physical cores, no CUDA, and a system drive
with limited headroom. `scripts/dev-env.*` set these in the environment before
anything downloads; the values here are the fallback for someone who skipped that
step, chosen to be safe rather than fast.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# usecwd=True so that an installed copy run from another directory does not pick
# up this repository's .env.
load_dotenv(find_dotenv(usecwd=True), override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Storage ---------------------------------------------------------------
    data_dir: str = ""  # INDICRAG_DATA_DIR; falls back to ./data

    # --- Encoders --------------------------------------------------------------
    encoder_primary: str = "intfloat/multilingual-e5-base"
    encoder_batch_size: int = 16
    encoder_max_seq_length: int = 320

    # --- Retrieval -------------------------------------------------------------
    retrieval_top_k: int = 5
    retrieval_candidates: int = 50
    hybrid_alpha: float = 0.4
    hybrid_method: str = "rrf"  # "rrf" | "weighted"
    rrf_k: int = 60
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Generation ------------------------------------------------------------
    llm_backend: str = "extractive"  # "llamacpp" | "openai" | "extractive"
    llm_gguf_path: str = ""
    llm_model: str = ""
    llm_n_ctx: int = 4096
    llm_n_threads: int = 4
    llm_max_tokens: int = 320
    llm_temperature: float = 0.0
    llm_seed: int = 20260922
    llm_base_url: str = ""
    llm_api_key: str = ""

    # --- Answerability ---------------------------------------------------------
    answerability_mode: str = "threshold"  # threshold | selfreport | nli | calibrated
    answerability_tau: float = 0.35
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    # --- Evaluation ------------------------------------------------------------
    rng_seed: int = 20260922
    bootstrap_resamples: int = 1000

    # --- Derived paths ---------------------------------------------------------

    @property
    def root(self) -> Path:
        """Data root. Honours INDICRAG_DATA_DIR, else ./data beside the project."""
        raw = self.data_dir or os.environ.get("INDICRAG_DATA_DIR") or "data"
        p = Path(raw).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _sub(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw_dir(self) -> Path:
        return self._sub("raw")

    @property
    def text_dir(self) -> Path:
        return self._sub("text")

    @property
    def emb_dir(self) -> Path:
        return self._sub("emb")

    @property
    def lex_dir(self) -> Path:
        return self._sub("lex")

    @property
    def gen_dir(self) -> Path:
        return self._sub("gen")

    @property
    def passages_path(self) -> Path:
        return self.root / "passages.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "corpus_manifest.jsonl"

    def emb_path(self, model: str) -> Path:
        return self.emb_dir / f"{slugify(model)}.npy"

    def emb_meta_path(self, model: str) -> Path:
        return self.emb_dir / f"{slugify(model)}.meta.json"


def slugify(model: str) -> str:
    """HF model id -> filename-safe slug. `intfloat/multilingual-e5-base` ->
    `intfloat__multilingual-e5-base`."""
    return model.replace("/", "__").replace(":", "_")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
