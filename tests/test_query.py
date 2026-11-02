"""The query path: the fixed and adaptive budgets are never exceeded, packing fills close to the budget, recall()
makes no LLM call, and each question makes exactly one LLM call across retries (the model call is monkeypatched;
no model is loaded)."""

from __future__ import annotations

import asyncio
import random

import httpx
import pytest

import _stubs as S
from spinalmemory import answer as A
from spinalmemory import llm
from spinalmemory.config import load_config
from spinalmemory.project.projection import full_build
from spinalmemory.query import budget as B
from spinalmemory.query import pack as P
from spinalmemory.query import retrieve as RT
from spinalmemory.truth import read_store, write_session
from spinalmemory.write import build as WB


def _scores(top: float, rest: float, n: int = 13) -> dict[str, float]:
    return {f"t{i}": s for i, s in enumerate([top] + [rest] * (n - 1))}


def _disk_store(tmp_path, seed: int):
    store = S.Store(random.Random(seed), "locomo")
    root = tmp_path / f"store{seed}"
    for sess in store.files.values():
        write_session(root, sess)
    (root / "index").mkdir()
    proj = full_build("locomo", read_store(root), S.STUB_CODECS, root / "index" / "fts.sqlite")
    WB.materialise_vectors(proj, root / "index", S.StubEncoder(), S.DictCache())
    proj.save(root / "index" / "projection.pkl")
    return root


def _answerer(cfg, monkeypatch):
    monkeypatch.setattr(RT, "rerank_scores", lambda name, dev, q, cands, ml=512: {
        c: round(float(len(text) % 11), 4) for c, text in cands})
    ans = A.Answerer(cfg)
    ans.encoder = S.StubEncoder()
    return ans


def test_fixed_and_adaptive_budget_choice():
    fixed = load_config("default", {"pack.policy": "fixed"})["pack"]
    assert B.choose(fixed, _scores(9.0, 1.0)) == (3000, 8.0) and B.choose(fixed, _scores(2.0, 2.0)) == (3000, 0.0)
    adaptive = load_config()["pack"]  # the default: 1,500 when the reranker is confident, else 3,000
    assert B.choose(adaptive, _scores(9.0, 1.0)) == (1500, 8.0)
    assert B.choose(adaptive, _scores(3.0, 1.0))[0] == 1500   # margin 2.0 >= 1.99
    assert B.choose(adaptive, _scores(2.98, 1.0))[0] == 3000  # margin 1.98 < 1.99
    assert B.choose(adaptive, {"a": 1.0})[0] == 1500          # a pool shorter than k counts as confident


@pytest.mark.parametrize("policy", ["fixed", "adaptive"])
def test_packed_memory_never_exceeds_the_budget(tmp_path, monkeypatch, policy):
    cfg = load_config("default", {"pack.policy": policy, "pack.budget": 60, "pack.budget_small": 30,
                                  "pack.margin_threshold": 5.0})
    ans = _answerer(cfg, monkeypatch)
    for seed in range(6):
        store = RT.OpenStore(str(_disk_store(tmp_path, seed)))
        r = ans.retrieve({"question_id": f"q{seed}", "question": "Where did Priya go for pottery?"}, store)
        assert r["memory_tokens"] <= r["budget"]
        assert r["budget"] == 60 if policy == "fixed" else r["budget"] in (30, 60)
        assert r["packed"] and set(r["packed"]) <= set(r["order"])


def test_greedy_packing_skips_what_does_not_fit():
    cost = {"a": 50, "b": 30, "c": 60, "d": 10}
    session = {"a": "s1", "b": "s1", "c": "s2", "d": "s2"}
    assert P.greedy(["a", "c", "b", "d"], cost, 99, session, {"s1": 5, "s2": 5}) == ["a", "b"]


def test_exact_fill_never_exceeds_and_lands_close():
    gaps = []
    for seed in range(100):
        rng = random.Random(seed)
        order = [f"t{i}" for i in range(rng.randint(5, 120))]
        cost = {t: rng.randint(5, 200) for t in order}
        session = {t: f"s{rng.randint(0, 9)}" for t in order}
        headers = {f"s{i}": 6 for i in range(10)}
        budget, extra = rng.choice([500, 1500, 3000]), rng.choice([0, 3, 12, 30])

        def measure(packed):  # a section outside the plan that grows with the packed turns
            return "memory", P.planned(packed, cost, session, headers) + extra * min(len(packed), 25)

        packed, _, tok, _ = P.fill(order, cost, budget, session, headers, measure)
        assert tok <= budget and tok == measure(packed)[1]
        if P.planned(order, cost, session, headers) > budget:  # the budget binds
            gaps.append((budget - tok) / budget)
    assert sum(gaps) / len(gaps) < 0.02  # on average under 2% of the budget is left unused
    # nothing planned fits: trimmed down to an empty block
    order = [f"t{i}" for i in range(10)]
    packed, _, tok, _ = P.fill(order, {t: 10 for t in order}, 50, {t: "s" for t in order}, {"s": 1},
                               lambda p: (None, 95 + 10 * len(p) if p else 0))
    assert packed == [] and tok == 0


def test_exactly_one_llm_call_per_question_across_retries(tmp_path, monkeypatch):
    cfg = load_config()
    ans = _answerer(cfg, monkeypatch)
    handle = WB.StoreHandle(str(_disk_store(tmp_path, 1)), "locomo", "digest")
    calls, outcomes = [], []

    async def fake_chat(prompt, **kwargs):
        calls.append(prompt)
        exc = outcomes.pop(0) if outcomes else None
        if exc:
            raise exc
        return "Answer: Leeds [1]\nEvidence: [1]", {"models": {"stub": {"calls": 1, "completion": 5}}}

    monkeypatch.setattr(llm, "chat", fake_chat)

    question = {"question_id": "q1", "question": "Where does Tom live?"}
    out = asyncio.run(ans.answer(question, handle))
    assert out["hypothesis"] == "Leeds" and out["extra"]["calls"] == 1 and len(calls) == 1
    with pytest.raises(A.OneCallViolation):  # a retry after a successful call never calls again
        asyncio.run(ans.answer(question, handle))
    assert len(calls) == 1

    # the request may have reached the API (read timeout): the retry fails instead of sending a second one
    outcomes[:] = [httpx.ReadTimeout("slow")]
    q2 = {"question_id": "q2", "question": "What does Priya do?"}
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(ans.answer(q2, handle))
    with pytest.raises(A.OneCallViolation):
        asyncio.run(ans.answer(q2, handle))
    assert len(calls) == 2

    # the request never left the process (connection refused): one retry may send it
    outcomes[:] = [httpx.ConnectError("refused")]
    q3 = {"question_id": "q3", "question": "What does Priya do?"}
    with pytest.raises(httpx.ConnectError):
        asyncio.run(ans.answer(q3, handle))
    assert asyncio.run(ans.answer(q3, handle))["extra"]["calls"] == 1
    assert ans.sent == {"q1": 1, "q2": 1, "q3": 1}


def test_recall_makes_no_llm_call_and_answer_makes_one(tmp_path, monkeypatch):
    ans = _answerer(load_config(), monkeypatch)
    handle = WB.StoreHandle(str(_disk_store(tmp_path, 2)), "locomo", "digest")
    calls = []

    async def no_call(prompt, **kwargs):
        calls.append(prompt)
        raise AssertionError("recall() called the LLM")

    monkeypatch.setattr(llm, "chat", no_call)
    memory = ans.recall("Where did Priya go for pottery?", handle)
    assert calls == [] and memory.budget in (1500, 3000) and 0 < memory.tokens <= memory.budget
    assert memory.tokens == llm.count_tokens(memory.context) and memory.context in memory.prompt
    assert [p.handle for p in memory.passages] == list(range(1, len(memory.passages) + 1))

    async def one_call(prompt, **kwargs):
        calls.append(prompt)
        return "Answer: Leeds [1]\nEvidence: [1]", {"models": {"stub": {"calls": 1, "completion": 5}}}

    monkeypatch.setattr(llm, "chat", one_call)
    out = asyncio.run(ans.answer({"question_id": "q9", "question": "Where did Priya go for pottery?"}, handle))
    assert calls == [memory.prompt] and out["extra"]["calls"] == 1
    assert out["extra"]["budget"] == memory.budget and out["extra"]["memory_tokens"] == memory.tokens
