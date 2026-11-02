"""Vectors: bge fp32 embeddings cached by sha256(model revision, text), plus the frozen cross-encoders.

Both are encoders (no text is generated). Determinism: fp32, pinned revision, one device per run, inputs scored
in a canonical order (text length, then text) in fixed batches. Device "auto" picks cuda, then mps, then cpu.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from spinalmemory.paths import VECTORS

# Pinned model revisions.
REVISIONS = {
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "BAAI/bge-small-en-v1.5": None,
    "mixedbread-ai/mxbai-rerank-base-v1": "800f24c113213a187e65bde9db00c15a2bb12738",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "mixedbread-ai/mxbai-rerank-xsmall-v1": "b5c6e9da73abc3711f593f705371cdbe9e0fe422",
}
RERANKERS = {
    "mxbai": "mixedbread-ai/mxbai-rerank-base-v1",
    "minilm": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "mxbai-xsmall": "mixedbread-ai/mxbai-rerank-xsmall-v1",
}
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
BATCH = 32
GPU_LOCK = threading.Lock()  # one encoder call at a time per process (also keeps timings honest)


def resolve_device(device: str) -> str:
    """"auto" -> the first of cuda, mps, cpu that is available; any other value is returned unchanged."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encoder_id(model: str) -> str:
    return f"{model}@{REVISIONS.get(model) or 'main'}"


@lru_cache(maxsize=4)
def _bi_encoder(model: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return SentenceTransformer(model, device=resolve_device(device), revision=REVISIONS.get(model), model_kwargs={"torch_dtype": torch.float32})


@lru_cache(maxsize=8)
def _cross_encoder(name: str, device: str, max_length: int = 512):
    import torch
    from sentence_transformers import CrossEncoder

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = RERANKERS[name]
    return CrossEncoder(model, device=resolve_device(device), max_length=max_length, revision=REVISIONS.get(model),
                        model_kwargs={"torch_dtype": torch.float32}, activation_fn=torch.nn.Identity())


class VectorCache:
    """sha256(model revision, text) -> fp32 vector, in one SQLite file shared by every store and process."""

    def __init__(self, path: Path = VECTORS):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=120, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, data BLOB)")
        self.db.commit()
        self.lock = threading.Lock()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self.lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT key, data FROM vec WHERE key IN ({','.join('?' * len(chunk))})"
                for k, blob in self.db.execute(q, chunk):
                    out[k] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        with self.lock:
            self.db.executemany("INSERT OR IGNORE INTO vec(key, data) VALUES (?, ?)",
                                [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()])
            self.db.commit()


class Encoder:
    """The frozen bi-encoder for passages (index keys) and queries."""

    def __init__(self, model: str, device: str):
        self.model, self.device = model, device
        self.id = encoder_id(model)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), np.float32)
        m = _bi_encoder(self.model, self.device)
        order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), texts[i]))
        with GPU_LOCK:
            v = m.encode([texts[i] for i in order], batch_size=BATCH, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        out = np.empty_like(v)
        out[order] = v
        return out

    def encode_query(self, text: str) -> np.ndarray:
        m = _bi_encoder(self.model, self.device)
        with GPU_LOCK:
            return m.encode([text], batch_size=1, prompt=BGE_QUERY_PROMPT, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)[0]


def rerank_scores(name: str, device: str, question: str, cands: list[tuple[str, str]],
                  max_length: int = 512) -> dict[str, float]:
    """Cross-encoder scores for (id, text) candidates, rounded to 1e-4 (the caller breaks ties).

    Batches follow (text length, text); the stable sort keeps the caller's order for identical texts, so batch
    composition never depends on how ids are spelled."""
    if not cands:
        return {}
    ce = _cross_encoder(name, device, max_length)
    order = sorted(cands, key=lambda c: (len(c[1]), c[1]))
    with GPU_LOCK:
        s = ce.predict([(question, text) for _, text in order], batch_size=BATCH, show_progress_bar=False, convert_to_numpy=True)
    return {cid: round(float(x), 4) for (cid, _), x in zip(order, np.asarray(s, dtype=np.float32))}


def warm(cfg: dict) -> None:
    """Load the encoders a config needs (outside any timed region)."""
    dev = cfg["device"]
    _bi_encoder(cfg["build"]["encoder"], dev["encoder"])
    _cross_encoder(cfg["rank"]["scorer"], dev["reranker"], dev["reranker_max_length"])
