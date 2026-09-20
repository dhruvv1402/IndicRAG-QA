"""Dense retrieval: encode, cache, and search exactly.

**No approximate index.** At ~700 passages, embeddings fit in a single float32
matrix and search is one matrix-vector product followed by `argpartition` -- under
a millisecond, and *exactly* correct. An ANN index at this scale would introduce
recall error indistinguishable from model error in the results, which is the one
thing this project cannot afford, since telling those apart is its whole purpose.
FAISS becomes the right answer somewhere past ~10^5 passages, and not before.

The cache invariant is the part worth reading twice. An embedding matrix silently
misaligned with the passage list it was built from produces plausible, ordered,
entirely meaningless numbers, and nothing about the output looks wrong. Every
cache therefore stores a hash of the passages it was built from and **refuses to
load** when that hash differs, rather than trusting the file name.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import Passage, Retrieved
from .encoders import EncoderSpec, get


class CacheMismatch(RuntimeError):
    """Raised when a cached matrix does not match the current passage set."""


def passages_fingerprint(passages: Sequence[Passage]) -> str:
    """Hash the passage IDs and text that an embedding matrix was built from."""
    import hashlib

    h = hashlib.sha256()
    for p in passages:
        h.update(p.passage_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(p.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


@dataclass
class DenseIndex:
    spec: EncoderSpec
    passage_ids: list[str]
    matrix: np.ndarray  # (n, dim), L2-normalized float32
    fingerprint: str
    pooling: str = "mean"

    @property
    def key(self) -> str:
        """Cache key. Includes pooling for MLM checkpoints, which are encoded
        twice -- mean and CLS -- so the H6 result cannot be written off as a
        pooling artefact. Without this the two variants overwrite each other."""
        if self.spec.is_mlm:
            return f"{self.spec.slug}.{self.pooling}"
        return self.spec.slug

    @staticmethod
    def cache_key(spec: EncoderSpec, pooling: str) -> str:
        return f"{spec.slug}.{pooling}" if spec.is_mlm else spec.slug

    # --- search ---------------------------------------------------------------

    def search_vector(self, vec: np.ndarray, k: int = 10) -> list[Retrieved]:
        """Exact cosine search. `vec` must already be L2-normalized."""
        scores = self.matrix @ vec.astype(np.float32)
        k = min(k, len(scores))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        method = f"dense:{self.spec.slug}"
        return [
            Retrieved(
                passage_id=self.passage_ids[int(i)],
                score=float(scores[int(i)]),
                rank=rank,
                method=method,
                component_scores={"dense": float(scores[int(i)])},
            )
            for rank, i in enumerate(top, start=1)
        ]

    # --- persistence ----------------------------------------------------------

    def save(self, emb_dir: Path) -> None:
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_dir / f"{self.key}.npy", self.matrix)
        (emb_dir / f"{self.key}.meta.json").write_text(
            json.dumps(
                {
                    "model": self.spec.name,
                    "pooling": self.pooling,
                    "n_passages": len(self.passage_ids),
                    "dim": int(self.matrix.shape[1]),
                    "fingerprint": self.fingerprint,
                    "passage_ids": self.passage_ids,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, spec: EncoderSpec, emb_dir: Path, passages: Sequence[Passage], pooling: str = "mean"
    ) -> DenseIndex:
        key = cls.cache_key(spec, pooling)
        meta_path = Path(emb_dir) / f"{key}.meta.json"
        npy_path = Path(emb_dir) / f"{key}.npy"
        if not (meta_path.exists() and npy_path.exists()):
            raise FileNotFoundError(f"no cached embeddings for {spec.name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        want = passages_fingerprint(passages)
        if meta.get("fingerprint") != want:
            raise CacheMismatch(
                f"cached embeddings for {spec.name} were built from a different passage set "
                f"({meta.get('fingerprint', '?')[:12]} != {want[:12]}); rebuild the index"
            )
        return cls(
            spec=spec,
            passage_ids=list(meta["passage_ids"]),
            matrix=np.load(npy_path),
            fingerprint=meta["fingerprint"],
            pooling=meta.get("pooling", "mean"),
        )


# --- encoding ------------------------------------------------------------------


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class Encoder:
    """Wraps either a SentenceTransformer or a raw MLM checkpoint behind one call.

    The MLM path is hand-rolled because `SentenceTransformer` would silently add
    an untrained mean-pooling head and present the result as a sentence encoder.
    Doing the pooling explicitly here keeps the H6 comparison honest and makes the
    CLS-vs-mean ablation possible at all.
    """

    def __init__(self, spec: EncoderSpec, pooling: str | None = None, max_seq_length: int = 320):
        self.spec = spec
        self.pooling = pooling or spec.pooling
        self.max_seq_length = max_seq_length
        self._st = None
        self._tok = None
        self._model = None

        if spec.is_mlm:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(spec.name)
            self._model = AutoModel.from_pretrained(spec.name)
            self._model.eval()
        else:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(spec.name, device="cpu")
            self._st.max_seq_length = max_seq_length

    def _encode_mlm(self, texts: list[str], batch_size: int) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                hidden = self._model(**enc).last_hidden_state
                if self.pooling == "cls":
                    vec = hidden[:, 0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(vec.cpu().numpy())
        return np.vstack(out)

    def encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        batch_size: int = 16,
        progress: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        prefix = self.spec.query_prefix if is_query else self.spec.passage_prefix
        prepared = [prefix + t for t in texts] if prefix else list(texts)

        if self._st is not None:
            arr = self._st.encode(
                prepared,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
        else:
            arr = self._encode_mlm(prepared, batch_size)

        if progress:
            progress(f"encoded {len(prepared)} texts with {self.spec.name} ({self.pooling})")
        return _normalize(np.asarray(arr, dtype=np.float32))

    def encode_query(self, text: str, batch_size: int = 1) -> np.ndarray:
        return self.encode([text], is_query=True, batch_size=batch_size)[0]


def build_dense(
    passages: Sequence[Passage],
    model: str,
    emb_dir: Path,
    *,
    pooling: str | None = None,
    batch_size: int = 16,
    progress: Callable[[str], None] | None = None,
) -> tuple[DenseIndex, float]:
    """Encode every passage and cache the matrix. Returns (index, seconds)."""
    spec = get(model)
    say = progress or (lambda _m: None)
    encoder = Encoder(spec, pooling=pooling)

    start = time.time()
    matrix = encoder.encode(
        [p.text for p in passages], is_query=False, batch_size=batch_size, progress=None
    )
    elapsed = time.time() - start

    index = DenseIndex(
        spec=spec,
        passage_ids=[p.passage_id for p in passages],
        matrix=matrix,
        fingerprint=passages_fingerprint(passages),
        pooling=encoder.pooling,
    )
    index.save(Path(emb_dir))
    say(
        f"  {spec.name:58s} {matrix.shape[0]:4d}x{matrix.shape[1]:<4d} "
        f"{elapsed:6.1f}s  {len(passages)/max(elapsed,1e-6):5.1f} passages/s"
    )
    return index, elapsed
