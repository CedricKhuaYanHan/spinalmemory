"""The query path for one question: search, fact history, rerank, pack and render by code, then exactly one
answer call.

    ans = Answerer(cfg)
    memory = ans.recall(question, handle)          # the packed memory, no LLM call
    result = await ans.answer(question, handle)    # recall, then one answer call (llm.settings(cfg, "answer"))

Search (BM25 + cosine + the named persons' turns, RRF) -> check fact history (attribute spines of the matching
persons; their source turns are added to the pool, query/history.py) -> one pool of `rank.limit` turns -> one
cross-encoder -> the size policy (query/budget.py) -> render the evidence (fact history, passages with dates,
fact lines with valid dates and corrections; query/render.py). The temporal path (question-date brackets, Notes,
320 output tokens) applies when the question carries its own date and matches the temporal rule; the advice
path when the question matches the advice rule. Neither changes the budget.

Retrieval (question in -> prompt ready) runs under a process-wide lock, so concurrent questions do not inflate
each other's retrieval_s; the answer call runs outside it. Opening a store and loading the encoders happen before
the timer starts and are reported apart.

One call per question is enforced per question across attempts: a caller may retry answer() after an
exception, so the count of generative requests is kept by question id across attempts, and a retry after a
request that may have reached the API fails the question instead of sending a second one. Nothing after the
answer call raises.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

import httpx

from spinalmemory import llm
from spinalmemory.project.vectors import Encoder, warm
from spinalmemory.query import budget as B
from spinalmemory.query import history as H
from spinalmemory.query import pack as P
from spinalmemory.query import render as R
from spinalmemory.query import temporal as TQ
from spinalmemory.query.parse import parse_question
from spinalmemory.query.retrieve import ce_order, first_stage, open_store
from spinalmemory.write.build import build_activity

RETRIEVAL_LOCK = threading.Lock()

# Errors raised before the request left this process: the API never saw it, so it is not a generative call.
NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout)


class OneCallViolation(RuntimeError):
    pass


class OneCall:
    """The instrumented answer client for one question. `sent` maps question id -> requests that may have
    reached the API, shared across retries of answer(); a second request is refused."""

    def __init__(self, sent: dict[str, int], qid: str):
        self.sent, self.qid = sent, qid

    @property
    def calls(self) -> int:
        return self.sent.get(self.qid, 0)

    async def chat(self, prompt: str, **kwargs):
        if self.calls >= 1:
            raise OneCallViolation(f"question {self.qid}: a generative request was already sent")
        self.sent[self.qid] = 1
        try:
            return await llm.chat(prompt, **kwargs)
        except NOT_SENT:
            self.sent[self.qid] = 0
            raise


@dataclass(frozen=True)
class Passage:
    """A packed turn, in reading order. `handle` is its [n] in the context."""

    id: str
    handle: int
    session: str
    date: str      # the session date, ISO
    speaker: str
    text: str


@dataclass(frozen=True)
class Fact:
    """A fact line citing a packed turn (the context shows the human ones, render.facts = "added")."""

    id: str
    turn: str                  # the turn the line sits under
    text: str
    subject: str
    attribute: str
    value: str
    valid: str | None          # the date it applies to, when known
    learned: str               # the date it was said, ISO
    human: bool
    corrected: bool            # a human line that corrects a machine line
    replaced: tuple[str, ...]  # the values it corrected


@dataclass(frozen=True)
class Recall:
    """The packed memory for one question. `context` is the memory block (`tokens` budget tokens, at most
    `budget`); `prompt` and `system` are the full request the built-in answer call sends."""

    question: str
    context: str
    tokens: int
    budget: int
    passages: list[Passage]
    facts: list[Fact]
    prompt: str
    system: str
    details: dict = field(repr=False)  # diagnostics: retrieval order, timings, history, flags


def _as_question(question: dict | str) -> dict:
    return {"question": question} if isinstance(question, str) else question


def _recall(r: dict, store) -> Recall:
    proj = store.proj
    passages = [Passage(t, h, proj.turns[t]["session"], proj.sessions[proj.turns[t]["session"]]["date"],
                        proj.turns[t]["speaker"], proj.turns[t]["text"]) for h, t in sorted(r["handles"].items())]
    facts = []
    for f in dict.fromkeys(f for p in passages for f in proj.turn_facts(p.id)):
        fr = proj.facts[f]
        replaced = tuple(proj.facts[m]["value"] for m in proj.corrects.get(f, ()))
        facts.append(Fact(f, fr["home"], fr["sentence"], fr["subject"], fr["attribute"], fr["value"],
                          proj.valid_text(f), fr["said_at"][:10], fr["tag"] == "h", bool(replaced), replaced))
    return Recall(r["question"], r["memory"], r["memory_tokens"], r["budget"], passages, facts, r["prompt"],
                  r["system"], r)


class Answerer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.encoder = Encoder(cfg["build"]["encoder"], cfg["device"]["encoder"])
        self.sent: dict[str, int] = {}  # question id -> generative requests sent (across retries)

    def warm(self) -> None:
        """Load the encoder, the cross-encoder and the budget tokenizer before any timed retrieval."""
        warm(self.cfg)
        llm.count_tokens("warm")
        self.encoder.encode_query("warm")

    # ---------------------------------------------------------------- retrieval (no generative model)

    def retrieve(self, question: dict, store) -> dict:
        """Question in -> packed prompt out. The only branches are on the input format (store.has_roles, the
        question's own date, the persons it names) and on the question text."""
        cfg = self.cfg
        dev, max_len = cfg["device"]["reranker"], cfg["device"]["reranker_max_length"]
        sc, hc, rk, pk, rc = cfg["search"], cfg["history"], cfg["rank"], cfg["pack"], cfg["render"]
        proj = store.proj
        times = {}
        t0 = time.perf_counter()
        q = parse_question(question, store, rc["count_items_only"])                             # parse
        kind = q.temporal if cfg["temporal"]["enabled"] else None                               # temporal path
        advice = bool(q.advice and cfg["advice"]["enabled"] and not kind)                       # advice path
        qvec = self.encoder.encode_query(q.retrieval_text)                                      # encode
        times["parse_encode_s"] = time.perf_counter() - t0
        first = first_stage(store, q.partition, q.retrieval_text, qvec, q.named, sc["rrf_k"],   # search
                            bool(sc["person_list"]))
        times["first_stage_s"] = time.perf_counter() - t0 - times["parse_encode_s"]
        part_set = set(store.parts[q.partition])
        th = time.perf_counter()
        spines, additions, hstats = [], [], {}
        if hc["enabled"]:                                                                       # fact history
            spines, additions, hstats = H.check(store, first, q.named, part_set, hc["seed_k"], hc["min_members"])
        times["history_s"] = time.perf_counter() - th
        t1 = time.perf_counter()
        pool, adds = H.one_pool(first, additions, int(rk["limit"]), int(hc["max_additions"]))  # one pool
        ce, scores = ce_order(store, q.retrieval_text, pool, rk["scorer"], dev, max_len)
        in_pool = set(pool)
        base = ce + [t for t in first if t not in in_pool]
        times["rerank_s"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        budget, margin = B.choose(pk, scores)                                                   # size policy
        session = {t: proj.turns[t]["session"] for t in base}
        sess_turns = {}

        def session_of(t):
            sid = proj.turns[t]["session"]
            if sid not in sess_turns:
                sess_turns[sid] = [x for x in proj.session_turns(sid) if x in part_set]
            return sess_turns[sid]

        order = base
        if pk["episode"]:
            order = P.episode_complete(order, session_of, proj.costs, pk["episode_seeds"], pk["episode_prev"],
                                       pk["episode_next"], pk["episode_cap"])
        facts = R.facts_mode(rc["facts"])  # True, False or "added" (only human / corrected fact lines)
        if facts == "added":
            cost = {t: proj.costs[t] + R.added_fact_cost(proj, t, llm.count_tokens) for t in order}
        else:
            cost = {t: proj.costs[t] + (proj.fact_costs.get(t, 0) if facts else 0) for t in order}
        headers = {s: llm.count_tokens(R.session_header(proj.sessions[s]["date"], q.own_date, q.question_date)) + 2
                   for s in {session[t] for t in order}}
        shown_spines = spines if rc["history_block"] else []
        qd_brackets = bool(kind and cfg["temporal"]["brackets"])  # temporal path: question-date brackets

        def measure(turns):                                                                     # render, counted
            rendered = R.render_memory(store, turns, q.question_date, rc["template_dates"], q.own_date, facts,
                                       shown_spines, rc["history_values"], qd_brackets)
            return rendered, llm.count_tokens(rendered[0])

        packed, (memory, handles, info), mem_tokens, fill_rounds = P.fill(order, cost, budget, session, headers,
                                                                          measure)
        planned_tokens = P.planned(packed, cost, session, headers)
        date_line = R.question_date_line(q.own_date, q.question_date_iso)
        qtimes = []
        if kind:                                                                                # temporal path
            qtimes = TQ.question_times(q.text, q.question_date)
            prompt = TQ.render_prompt(q.text, memory, date_line, kind, TQ.time_line(qtimes, q.question_date),
                                      R.intro_notes(info), q.counting)
        else:
            prompt = R.render_prompt(q.text, memory, date_line, R.intro_notes(info), counting=q.counting,
                                     advice=advice)
        times["pack_render_s"] = time.perf_counter() - t2
        packed_set = set(packed)
        return {"question": q.text, "memory": memory, "prompt": prompt, "system": R.SYSTEM, "packed": packed,
                "handles": handles, "order": order,
                "memory_tokens": mem_tokens, "planned_tokens": planned_tokens, "fill_rounds": fill_rounds,
                "partition": q.partition, "named": q.named, "pairs": len(pool),
                "question_date": q.question_date, "own_date": q.own_date, "budget": budget,
                "budget_policy": pk["policy"], "budget_margin": None if margin is None else round(margin, 4),
                "temporal": kind,
                "question_times": [{"text": x["text"], "lo": x["lo"].isoformat(), "hi": x["hi"].isoformat(),
                                    "name": x["name"]} for x in qtimes],
                "advice": advice, "counting": q.counting,
                "history": {"spines": [list(k) for k in spines], "additions": len(adds),
                            "additions_found": len(additions), "additions_packed": sum(t in packed_set for t in adds),
                            "pool_size": len(pool), "lines": info["history_lines"], **hstats},
                "fact_lines": info["fact_lines"], "corrected_shown": info["corrected_shown"],
                "max_tokens": cfg["temporal"]["max_tokens"] if kind else cfg["reader"]["max_tokens"],
                "times": {k: round(v, 4) for k, v in times.items()}}

    def recall(self, question: dict | str, handle) -> Recall:
        """The packed memory for a question (a string, or a dict as in the README), with no LLM call."""
        question = _as_question(question)
        t_open = time.perf_counter()
        store = open_store(handle.store_dir)
        open_s = time.perf_counter() - t_open
        with RETRIEVAL_LOCK:
            active0, started0 = build_activity()
            t0 = time.perf_counter()
            r = self.retrieve(question, store)
            r["retrieval_s"] = time.perf_counter() - t0
            active1, started1 = build_activity()
        # a store build in this process competes for the GPU and the GIL: flag the timing
        r["build_overlap"] = bool(active0 or active1 or started1 != started0)
        r["open_s"] = open_s
        return _recall(r, store)

    # ---------------------------------------------------------------- the answer call

    async def answer(self, question: dict | str, handle) -> dict:
        """recall(), then exactly one answer call. Retries of the same question_id never send a second one."""
        question = _as_question(question)
        # a retry passes the same question dict, so its id() is the fallback key
        qid = str(question.get("question_id") or f"object-{id(question)}")
        if self.sent.get(qid, 0):  # a retry after the reader request was sent: fail rather than call again
            raise OneCallViolation(f"question {qid}: a generative request was already sent")
        model = llm.settings(self.cfg, "answer")
        r = (await asyncio.to_thread(self.recall, question, handle)).details
        open_s = r["open_s"]
        prompt_tokens = llm.count_prompt_tokens(r["prompt"], r["system"])
        client = OneCall(self.sent, qid)
        max_tokens = r["max_tokens"]
        text, usage = await client.chat(r["prompt"], settings=model, system=r["system"], max_tokens=max_tokens)
        # Everything below is code over the one output; it must not raise, or the runner would retry the question.
        post_error = None
        hyp, evidence_ids, had_answer, notes, parsed = text.strip(), [], False, "", {}
        completion = sum(int(s.get("completion", 0) or 0) for s in (usage or {}).get("models", {}).values())
        truncated = completion >= max_tokens
        try:
            ans = self.cfg["answer"]
            parsed = R.read_answer(text, truncated, ans["cut_handles"], ans["empty_when_cut"])
            hyp, ev, had_answer = parsed["hypothesis"], parsed["evidence"], parsed["answer_line"]
            notes = R.notes_of(text) if r["temporal"] else ""
            evidence_ids = [r["handles"][h] for h in ev if h in r["handles"]]
        except Exception as exc:  # recorded, never raised (see the module docstring)
            post_error = f"{type(exc).__name__}: {exc}"[:300]
        rerank_s = r["times"].get("rerank_s") or 0.0
        return {
            "hypothesis": hyp,
            "retrieved_ids": r["order"][:100],
            "usage": usage,
            "extra": {
                "retrieval_s": round(r["retrieval_s"], 4),
                "prompt_tokens": prompt_tokens,
                "calls": client.calls,
                "packed_ids": r["packed"],
                "config": self.cfg["name"],
                "memory_tokens": r["memory_tokens"],
                "planned_tokens": r.get("planned_tokens"),
                "fill_rounds": r.get("fill_rounds"),
                "evidence_ids": evidence_ids,
                "answer_line": had_answer,
                "partition": r["partition"],
                "named_speakers": r["named"],
                "ce_pairs": r["pairs"],
                "ce_pairs_per_s": round(r["pairs"] / rerank_s, 1) if rerank_s > 0 else None,
                "build_overlap": r["build_overlap"],
                "times": r["times"],
                "open_s": round(open_s, 4),
                "cached_tokens": sum(s.get("cached", 0) for s in (usage or {}).get("models", {}).values()),
                "raw_output": text,
                "post_error": post_error,
                "budget": r["budget"],
                "budget_policy": r["budget_policy"],
                "budget_margin": r["budget_margin"],
                "own_date": r["own_date"],
                "history": r["history"],
                "history_spines": r["history"]["spines"],
                "history_additions": r["history"]["additions"],
                "pool_size": r["history"]["pool_size"],
                "fact_lines": r["fact_lines"],
                "corrected_shown": r["corrected_shown"],
                "temporal": r["temporal"],
                "question_times": r["question_times"],
                "advice": r["advice"],
                "counting": r["counting"],
                "max_tokens": max_tokens,
                "truncated": truncated,
                "notes_text": notes,
                "handles_cut": parsed.get("handles_cut", []),
                "answer_emptied": parsed.get("emptied"),
            },
        }
