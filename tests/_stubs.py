"""Deterministic stand-ins for the parts that would need a model: the writer, the codecs, the encoder and the
vector cache, plus a seeded generator of truth-file edits for the rebuild tests."""

from __future__ import annotations

import copy
import hashlib
import random

import numpy as np

from spinalmemory.project.projection import Codecs, Projection
from spinalmemory.truth import FactLine, SessionFile, TurnBlock

STUB_CODECS = Codecs(count_tokens=lambda s: len(s.split()), tokenizer_id="stub-tokenizer", encoder_id="stub-encoder")

NAMES = ["Priya", "Tom", "Tommy", "Ana", "Omar", "Biscuit"]
PHRASES = ["yesterday", "last week", "next month", "two weeks ago", "last Friday", "in June", "last year", "on 7 May 2023"]
WORDS = ["pottery", "Paris", "dance", "studio", "library", "job", "beagle", "marathon", "painting", "camping",
         "guitar", "bakery", "Rome", "garden", "novel", "bike", "museum", "school"]
ATTRS = ["job", "lives_in", "partner", "hobby", "visited", "event", "alias", "other:misc"]


class StubEncoder:
    id = "stub-encoder"

    def __init__(self, *a):
        pass

    def encode_passages(self, texts):
        return np.array([[len(t) % 7, len(t) % 5, 1.0] for t in texts], dtype=np.float32)

    def encode_query(self, text):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class DictCache:
    def __init__(self):
        self.store: dict = {}

    def get_many(self, keys):
        return {k: self.store[k] for k in keys if k in self.store}

    def put_many(self, items):
        self.store.update(items)


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def stub_lines(turn: TurnBlock) -> list[FactLine]:
    """The stub writer: 0-3 machine lines, a deterministic function of the turn."""
    h = _h(turn.id, turn.speaker, turn.text)
    words = turn.text.split()
    out = []
    for k in range(h % 4):
        hk = _h(h, k)
        attr = ATTRS[hk % len(ATTRS)]
        value = words[hk % len(words)] if words else "thing"
        subject = turn.speaker if hk % 3 else NAMES[hk % len(NAMES)]
        if attr == "alias":
            subject, value = NAMES[hk % 3], NAMES[3 + hk % 3]
        phrase = next((p for p in PHRASES if p in turn.text), None)
        when = f'"{phrase}"' if phrase and hk % 2 else "-"
        out.append(FactLine("m", f"{subject} {attr} {value}.", subject, attr, value, when))
    return out


def reingest(sess: SessionFile) -> SessionFile:
    """Writer + commit with the stub writer: machine lines rewritten from the turns, human lines kept."""
    out = copy.deepcopy(sess)
    for t in out.turns:
        t.facts = stub_lines(t) + [f for f in t.facts if f.tag == "h"]
    return out


def random_text(rng: random.Random) -> str:
    parts = [rng.choice(WORDS) for _ in range(rng.randint(2, 7))]
    if rng.random() < 0.5:
        parts.insert(rng.randint(0, len(parts)), rng.choice(PHRASES))
    if rng.random() < 0.3:
        parts.append(rng.choice(NAMES))
    return " ".join(parts)


def random_date(rng: random.Random) -> str:
    return f"2023-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.choice(['00', '30'])}"


class Store:
    """An in-memory set of truth files with a fresh-id counter (ids are never reused or renumbered).
    `fmt` is the input format label: "locomo" (named speakers) or "longmemeval" (user/assistant roles)."""

    def __init__(self, rng: random.Random, fmt: str):
        self.rng, self.benchmark, self.next_id = rng, fmt, 0
        self.files: dict[str, SessionFile] = {}
        for _ in range(rng.randint(2, 4)):
            self.new_session()

    def fresh_turn(self, sid: str) -> TurnBlock:
        self.next_id += 1
        if self.benchmark == "locomo":
            return TurnBlock(f"{sid}:{self.next_id}", self.rng.choice(NAMES[:2]), "speaker", random_text(self.rng))
        role = self.rng.choice(["user", "user", "assistant"])
        return TurnBlock(f"{sid}_{self.next_id}", role.capitalize(), role, random_text(self.rng))

    def new_session(self) -> str:
        self.next_id += 1
        sid = f"S{self.next_id}"
        sess = SessionFile(sid, random_date(self.rng), "", [self.fresh_turn(sid) for _ in range(self.rng.randint(2, 6))])
        self.files[sid] = reingest(sess)
        return sid

    def all_turns(self) -> list[tuple[str, int]]:
        return [(sid, i) for sid in sorted(self.files) for i in range(len(self.files[sid].turns))]

    def all_facts(self) -> list[tuple[str, int, int]]:
        return [(sid, i, j) for sid, i in self.all_turns() for j in range(len(self.files[sid].turns[i].facts))]


OPS = ["insert_turn", "edit_text", "delete_turn", "add_human", "edit_line", "delete_line", "session_date",
       "rename_subject", "rename_speaker", "add_session", "delete_session", "cross_session_source",
       "correct_machine", "confirm_machine", "duplicate_human"]


def apply_op(store: Store, op: str) -> set[str]:
    """Mutate the store with one edit; returns the changed session ids. Turn-text edits re-ingest the session."""
    rng, files = store.rng, store.files
    turns = store.all_turns()
    if op == "add_session" or not turns:
        return {store.new_session()}
    sid, i = rng.choice(turns)
    sess = copy.deepcopy(files[sid])
    t = sess.turns[i]
    if op == "insert_turn":
        sess.turns.insert(rng.randint(0, len(sess.turns)), store.fresh_turn(sid))
        sess = reingest(sess)
    elif op == "edit_text":
        t.text = random_text(rng)
        sess = reingest(sess)
    elif op == "delete_turn":
        del sess.turns[i]
    elif op in ("add_human", "cross_session_source"):
        attr = rng.choice(ATTRS)
        subj, val = (rng.choice(NAMES[:3]), rng.choice(NAMES[3:])) if attr == "alias" else (rng.choice(NAMES), rng.choice(WORDS))
        when = rng.choice(["-", f'"{rng.choice(PHRASES)}"', "@2023-06-01", "@2022"])
        also = ()
        if op == "cross_session_source":
            osid, oi = rng.choice(turns)
            also = (files[osid].turns[oi].id,)
        t.facts.append(FactLine("h", f"{subj} {attr} {val} (human).", subj, attr, val, when, also))
    elif op == "edit_line" and t.facts:
        j = rng.randrange(len(t.facts))
        f = t.facts[j]
        t.facts[j] = FactLine(f.tag, f.sentence + " edited", f.subject, rng.choice(ATTRS), rng.choice(WORDS),
                              rng.choice(["-", f'"{rng.choice(PHRASES)}"', "@2023-03-03"]), f.also)
    elif op == "delete_line" and t.facts:
        del t.facts[rng.randrange(len(t.facts))]
    elif op == "session_date":
        sess.date = random_date(rng)
    elif op == "rename_subject" and t.facts:
        j = rng.randrange(len(t.facts))
        f = t.facts[j]
        t.facts[j] = FactLine(f.tag, f.sentence, rng.choice(NAMES), f.attribute, f.value, f.when, f.also)
    elif op == "rename_speaker" and store.benchmark == "locomo":
        t.speaker = rng.choice(NAMES[:3])
        sess = reingest(sess)
    elif op in ("correct_machine", "confirm_machine"):
        ms = [(s, i2, j) for s, i2, j in store.all_facts() if files[s].turns[i2].facts[j].tag == "m"]
        if not ms:
            return set()
        sid, i, j = rng.choice(ms)
        sess = copy.deepcopy(files[sid])
        f = sess.turns[i].facts[j]
        value = f.value.upper() if op == "confirm_machine" else rng.choice(WORDS) + " (h)"
        sess.turns[i].facts.append(FactLine("h", f"{f.subject} {f.attribute} {value} (human).", f.subject,
                                            f.attribute, value, rng.choice(["-", "@2023-02-02", f.when])))
    elif op == "duplicate_human":
        t.facts += [FactLine("h", "Duplicate line.", t.speaker, "hobby", "x", "-")] * 2
    elif op == "delete_session":
        del files[sid]
        return {sid}
    files[sid] = sess
    return {sid}


def cold_build(store: Store, order_seed: int | None = None) -> Projection:
    """A full rebuild from the files. With order_seed, files are fed one at a time in a shuffled order."""
    proj = Projection(store.benchmark, STUB_CODECS)
    if order_seed is None:
        proj.update(dict(store.files))
    else:
        sids = sorted(store.files)
        random.Random(order_seed).shuffle(sids)
        for sid in sids:
            proj.update({sid: store.files[sid]})
    return proj
