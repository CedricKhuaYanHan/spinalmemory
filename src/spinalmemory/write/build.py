"""The write path for one history: append, windows, writer, validate, commit, project.

    handle, usage = await build_store(history, cfg)

The store directory (<cache>/stores/<build key>/<history hash>/, config.build_key) holds:
    sessions/*.md            the Markdown truth files
    ingest-manifest.json     writer provider and model, prompt version, window hashes, validation stats (never projected)
    index/                   the projection: projection.pkl (state), fts.sqlite (FTS5 BM25),
                             vectors.npy + vector_ids.json (dense, rows in turn-id order), dump.sha256 (canonical dump)
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spinalmemory import llm
from spinalmemory.config import build_key
from spinalmemory.paths import STORES
from spinalmemory.project.projection import Codecs, Projection, full_build
from spinalmemory.project.vectors import Encoder, VectorCache
from spinalmemory.truth import read_store, write_session
from spinalmemory.write.append import history_sessions
from spinalmemory.write.commit import commit
from spinalmemory.write.windows import make_window
from spinalmemory.write.writer import write_window

PROJECT_LOCK = threading.Lock()
_ACTIVITY_LOCK = threading.Lock()
_activity = {"active": 0, "started": 0}  # store builds in this process (answer.py flags timings that overlap one)


def build_activity() -> tuple[int, int]:
    """(builds running now, builds started so far) in this process."""
    with _ACTIVITY_LOCK:
        return _activity["active"], _activity["started"]


def _activity_add(n: int) -> None:
    with _ACTIVITY_LOCK:
        _activity["active"] += n
        _activity["started"] += max(n, 0)


@dataclass
class StoreHandle:
    """A built store: small and picklable; the store is opened lazily at query time."""

    store_dir: str
    benchmark: str
    digest: str
    stats: dict = field(default_factory=dict)


def history_digest(history: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(history, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def codecs_for(cfg: dict) -> Codecs:
    return Codecs(count_tokens=llm.count_tokens, tokenizer_id=llm.reader_tokenizer_id(),
                  encoder_id=Encoder(cfg["build"]["encoder"], cfg["device"]["encoder"]).id)


def _add_usage(acc: dict, usage: dict) -> None:
    for model, slot in (usage or {}).get("models", {}).items():
        a = acc.setdefault(model, {"calls": 0, "prompt": 0, "cached": 0, "cache_write": 0, "completion": 0})
        for k in a:
            a[k] += int(slot.get(k, 0) or 0)


async def build_store(history: dict, cfg: dict, stores: Path = STORES) -> tuple[StoreHandle, dict]:
    _activity_add(1)
    try:
        return await _build_store(history, cfg, stores)
    finally:
        _activity_add(-1)


async def _build_store(history: dict, cfg: dict, stores: Path) -> tuple[StoreHandle, dict]:
    b = cfg["build"]
    bench = history["benchmark"]
    digest = history_digest(history)
    store = stores / build_key(cfg) / digest
    writer = llm.settings(cfg, "writer")
    t0 = time.time()
    previous = read_store(store) if (store / "sessions").exists() else {}

    sessions = history_sessions(history)
    tok = llm.tokenizer() if bench == "longmemeval" else None
    windows = [make_window(s, bench, b["lme_turn_max_tokens"], tok) for s in sessions]
    results = await asyncio.gather(*(write_window(w, b["writer_prompt"], b["writer_max_tokens"], b["max_lines_per_turn"],
                                                  writer) for w in windows))
    made, equivalent = {}, {}
    manifest = {"writer_provider": writer.provider, "writer_model": writer.model, "prompt_version": b["writer_prompt"],
                "benchmark": bench, "history_hash": digest, "windows": []}
    for sess, w, (val, rec, cached) in zip(sessions, windows, results):
        for a in rec["attempts"]:
            _add_usage(equivalent, a["usage"])
            if not cached:
                _add_usage(made, a["usage"])
        manifest["windows"].append({"session": sess.session_id, "window_hash": w.hash, "cache_key": rec.get("key"),
                                    "attempts": len(rec["attempts"]), "cached": cached, "ok": val.ok, "stats": val.stats})
        write_session(store, commit(sess, val.lines, previous.get(sess.session_id)))
    keep = {s.session_id for s in sessions}
    for p in (store / "sessions").glob("*.md"):
        if p.read_text(encoding="utf-8").split("\n", 2)[1].removeprefix("session: ") not in keep:
            p.unlink()
    write_s = time.time() - t0
    (store / "ingest-manifest.json").write_text(json.dumps(manifest, indent=1))

    t1 = time.time()
    proj_stats = await asyncio.to_thread(project_store, store, bench, cfg)
    stats = {"sessions": len(sessions), "windows_written": sum(not c for _, _, c in results),
             "writer_calls": sum(len(r["attempts"]) for _, r, c in results if not c),
             "writer_calls_equivalent": sum(len(r["attempts"]) for _, r, _ in results),
             "windows_failed_validation": sum(not v.ok for v, _, _ in results),
             "windows_reused": sum(bool(c) for _, _, c in results),
             # writer time phrases in fact sentences that the window does not contain, cut by code
             "time_phrases_cut": sum(v.stats.get("time_phrases_cut", 0) for v, _, _ in results),
             "time_lines_changed": sum(v.stats.get("time_lines_changed", 0) for v, _, _ in results),
             "time_lines_dropped": sum(v.stats.get("time_lines_dropped", 0) for v, _, _ in results),
             "writer_usage_equivalent": equivalent, "write_s": round(write_s, 2),
             "project_s": round(time.time() - t1, 2), **proj_stats}
    (store / "build-stats.json").write_text(json.dumps(stats, indent=1))
    return StoreHandle(str(store), bench, digest, stats), {"models": made}


def project_store(store: Path, bench: str, cfg: dict) -> dict:
    """Project a whole store (cold full build), then the dense vectors from the shared cache.

    One projection at a time per process (PROJECT_LOCK) and per store across processes (an exclusive file lock;
    OpenStore takes it shared). The index is built in index.tmp/ and renamed into place, so a crash never
    leaves a half-written index/."""
    with PROJECT_LOCK, open(store / ".project.lock", "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        codecs = codecs_for(cfg)
        index, tmp, old = store / "index", store / "index.tmp", store / "index.old"
        for d in (tmp, old):
            if d.exists():
                shutil.rmtree(d)
        tmp.mkdir(parents=True)
        files = read_store(store)
        t0 = time.time()
        proj = full_build(bench, files, codecs, tmp / "fts.sqlite")
        project_s = time.time() - t0
        n_new = materialise_vectors(proj, tmp, Encoder(cfg["build"]["encoder"], cfg["device"]["encoder"]))
        proj.save(tmp / "projection.pkl")  # Projection.load finds fts.sqlite next to the pickle
        (tmp / "dump.sha256").write_text(proj.dump_hash() + "\n")
        proj.db.close()
        if index.exists():
            index.rename(old)
        tmp.rename(index)
        shutil.rmtree(old, ignore_errors=True)
        return {"turns": len(proj.turns), "facts": len(proj.facts), "spines": len(proj.spines),
                "projection_s": round(project_s, 2), "vectors_encoded": n_new}


def materialise_vectors(proj: Projection, index: Path, encoder: Encoder, cache: VectorCache | None = None) -> int:
    """Rows in turn-id order; vectors come from the shared cache, and only unseen texts are encoded."""
    cache = cache or VectorCache()
    ids = sorted(proj.keys)
    vkeys = [proj.vec_keys[t] for t in ids]
    have = cache.get_many(sorted(set(vkeys)))
    todo = sorted({k: proj.keys[t] for t, k in zip(ids, vkeys) if k not in have}.items())
    if todo:
        vecs = encoder.encode_passages([text for _, text in todo])
        new = {k: v for (k, _), v in zip(todo, vecs)}
        cache.put_many(new)
        have.update(new)
    mat = np.stack([have[k] for k in vkeys]).astype(np.float32) if ids else np.zeros((0, 768), np.float32)
    np.save(index / "vectors.npy", mat)
    (index / "vector_ids.json").write_text(json.dumps(ids))
    return len(todo)
