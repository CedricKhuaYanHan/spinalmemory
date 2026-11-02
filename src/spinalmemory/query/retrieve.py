"""Encode the question, hybrid first stage over the fact-keyed turn index, cross-encoder rerank.

Every ordering is total: scores are rounded and ties break on a hash of the turn id (`tie_key`), so a ranking is
a pure function of the store, the question and the pinned models. The tie-break hashes the id, so
that how ids are spelled (which may encode where a turn came from) never decides a tie.
"""

from __future__ import annotations

import fcntl
import json
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from spinalmemory.project.projection import Projection, rowid_of
from spinalmemory.project.vectors import Encoder, rerank_scores
from spinalmemory.query.parse import fts_query, speaker_aliases


def tie_key(item_id: str) -> int:
    """The label-free tie-break for every ranking: sha256 of the id (the FTS5 rowid), never the id's spelling."""
    return rowid_of(item_id)


class OpenStore:
    """A store opened read-only for querying: the projection state, its FTS5 tables and the dense matrix.

    Opening takes a shared lock on the store's projection lock file, so it waits while a build (in this or
    another process) is rewriting the index and never reads a half-written one."""

    def __init__(self, store_dir: str):
        index = Path(store_dir) / "index"
        self.dir = store_dir
        with open(Path(store_dir) / ".project.lock", "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            self.proj = Projection.load(index / "projection.pkl", None, readonly=True)
            self.vec_ids = json.loads((index / "vector_ids.json").read_text())
            self.mat = np.load(index / "vectors.npy")
        self.row = {t: i for i, t in enumerate(self.vec_ids)}
        self.tb = {t: tie_key(t) for t in self.proj.turns}
        self.rowid_to_turn = dict(self.proj.rowids)
        # input format: does the history have user/assistant roles?
        self.has_roles = any(t["role"] in ("user", "assistant") for t in self.proj.turns.values())
        # The user partition follows the roles and ignores the store's label. The projection builds a `user` FTS table only
        # for user/assistant-format stores; for any other role history BM25 runs over the `all` table, restricted
        # to user turns.
        names = tuple(self.proj.partitions) + (("user",) if self.has_roles and "user" not in self.proj.partitions else ())
        self.parts = {p: self.proj.partition_ids(p) for p in names}
        self.fts = {p: p if p in self.proj.partitions else "all" for p in names}
        self.last_date = max(s["date"] for s in self.proj.sessions.values())
        self.speaker_aliases = speaker_aliases(self.proj)  # empty when no turn has role "speaker"
        self.lock = threading.Lock()  # sqlite connection shared across threads

    def bm25(self, part: str, question: str) -> list[str]:
        q = fts_query(question)
        if not q:
            return []
        table = self.fts[part]
        with self.lock:
            rows = self.proj.db.execute(f"SELECT rowid, bm25(fts_{table}) FROM fts_{table} WHERE fts_{table} MATCH ?", (q,)).fetchall()
        scored = [(round(s, 9), r) for r, s in rows]  # rowid = tie_key(turn id)
        ids = [self.rowid_to_turn[r] for _, r in sorted(scored)]
        if table != part:
            keep = set(self.parts[part])
            ids = [t for t in ids if t in keep]
        return ids

    def dense(self, part: str, qvec: np.ndarray) -> list[str]:
        ids = self.parts[part]
        sims = self.mat[[self.row[t] for t in ids]] @ qvec
        return [t for _, _, t in sorted(zip((-round(float(s), 6) for s in sims), (self.tb[t] for t in ids), ids))]


_OPEN: "OrderedDict[str, OpenStore]" = OrderedDict()
_OPEN_LOCK = threading.Lock()


def open_store(store_dir: str, keep: int = 16) -> OpenStore:
    with _OPEN_LOCK:
        if store_dir in _OPEN:
            _OPEN.move_to_end(store_dir)
            return _OPEN[store_dir]
    st = OpenStore(store_dir)
    with _OPEN_LOCK:
        _OPEN[store_dir] = st
        while len(_OPEN) > keep:
            _OPEN.popitem(last=False)
    return st


def rrf(lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion; ties break on tie_key."""
    s: dict[str, float] = defaultdict(float)
    for rl in lists:
        for r, x in enumerate(rl):
            s[x] += 1.0 / (k + r + 1)
    return [x for x, _ in sorted(s.items(), key=lambda kv: (-round(kv[1], 12), tie_key(kv[0])))]


def first_stage(store: OpenStore, part: str, question: str, qvec: np.ndarray, named: list[str], k: int,
                speaker_boost: bool) -> list[str]:
    """First stage: FTS5 BM25 + exact cosine over the routed partition (+ the named persons' turns), RRF; the full order."""
    bm = store.bm25(part, question)
    dn = store.dense(part, qvec)
    base = rrf([bm, dn], k)
    if speaker_boost and named:
        spk = [t for t in base if store.proj.turns[t]["speaker"] in named]
        return rrf([bm, dn, spk], k)
    return base


def ce_order(store: OpenStore, question: str, ids: list[str], model: str, device: str,
             max_length: int = 512) -> tuple[list[str], dict]:
    """Rerank `ids` by a frozen cross-encoder over (question, index key); score rounded, ties on tie_key."""
    scores = rerank_scores(model, device, question, [(t, store.proj.keys[t]) for t in ids], max_length)
    return sorted(ids, key=lambda t: (-scores[t], store.tb[t])), scores


def encode_query(encoder: Encoder, question: str) -> np.ndarray:
    return encoder.encode_query(question)
