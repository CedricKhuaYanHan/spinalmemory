"""Rebuilds are a pure function of the truth files: the same files give the same canonical dump whether built
twice, fed in a shuffled order, read back from disk, or reached by incremental updates after random edits."""

from __future__ import annotations

import random

import pytest

import _stubs as S
from spinalmemory.project.projection import full_build
from spinalmemory.truth import read_store, write_session


def _store(seed: int) -> S.Store:
    return S.Store(random.Random(seed), "locomo" if seed % 2 else "longmemeval")


def test_same_files_rebuild_identically():
    for seed in range(20):
        store = _store(seed)
        assert S.cold_build(store).dump() == S.cold_build(store).dump()


def test_shuffled_file_order_gives_the_same_dump():
    for seed in range(20):
        store = _store(seed)
        ref = S.cold_build(store).dump()
        for order in range(5):
            assert S.cold_build(store, order_seed=order).dump() == ref


def test_build_from_disk_equals_build_in_memory(tmp_path):
    for seed in range(6):
        store = _store(seed)
        root = tmp_path / f"s{seed}"
        for sess in store.files.values():
            write_session(root, sess)
        on_disk = full_build(store.benchmark, read_store(root), S.STUB_CODECS, root / "fts.sqlite")
        assert on_disk.dump() == S.cold_build(store).dump()


@pytest.mark.parametrize("op", S.OPS)
def test_each_edit_kind_alone(op):
    for seed in range(15):
        store = S.Store(random.Random(1000 + seed), "locomo" if seed % 2 else "longmemeval")
        proj = S.cold_build(store)
        changed = S.apply_op(store, op)
        proj.update({sid: store.files.get(sid) for sid in changed})
        assert proj.dump() == S.cold_build(store).dump(), (op, seed)


def test_quick_fuzz_of_edit_sequences():
    """100 seeded sequences of 1-6 edits, some applied in batches before one update."""
    for seed in range(100):
        rng = random.Random(seed)
        store = S.Store(rng, "locomo" if seed % 2 else "longmemeval")
        proj = S.cold_build(store)
        for _ in range(rng.randint(1, 6)):
            changed: set[str] = set()
            for _ in range(rng.randint(2, 4) if seed % 3 == 2 else 1):
                changed |= S.apply_op(store, rng.choice(S.OPS))
            proj.update({sid: store.files.get(sid) for sid in sorted(changed)})
        cold = S.cold_build(store).dump()
        assert proj.dump() == cold, seed
        assert S.cold_build(store, order_seed=seed).dump() == cold, seed
