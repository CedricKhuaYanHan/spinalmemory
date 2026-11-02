"""The LLM-free projection of the Markdown truth files.

The projection is a pure function of the truth files and the pinned inputs (resolver, vocabulary, alias rules,
encoder id, budget tokenizer). There is one code path: `update(changes)` re-derives the changed turn blocks of
the changed session files plus their dependency closures, and a full build is `update` from an empty state with
every file. So any sequence of updates ends in the same canonical dump as a cold full build over the final files.

State: sessions and turns; facts and their source turns; entities and aliases; resolved time expressions;
attribute spines and correction links; per-session count keys; per-turn index key text, rendered costs and
vector keys; the FTS5 tables.

Dependency closures (what else an edit dirties):
- a changed block (text, speaker, caption, position, facts) -> its turn row, its time expressions, its facts;
- a session-date edit -> every block of that session (time expressions -> spine order, count keys);
- a fact added/removed/edited -> the index keys of every turn it cites (n:m, possibly in other sessions), its
  attribute spine, its session's count keys; an `alias` fact -> the alias union-find -> entity edges of every
  fact naming a changed name -> their spines and count keys; a name entering or leaving the entity set -> object
  edges of facts whose value is that name;
- a re-derived spine whose correction links changed -> the index keys and fact costs of every turn the
  corrected `m` line and its correcting `h` lines cite;
- a turn added/removed -> the fact-source edges of facts that cite it from elsewhere (`also:`).

Attribute spines. Key = (canonical subject entity, attribute) for every attribute except `alias`. Members are
ordered newest first by (valid date if known else learned date, learned date, hash of the fact id). The valid
date is when the fact applies (its `when` field, resolved by the date resolver; vague or absent = unknown, never
guessed); the learned date is when it was said (`said_at`). Single-valued attributes (vocab.SINGLE_VALUED) have a
current head; multi-valued ones are histories without a head. Correction rule: an `h` line with the same (entity,
attribute) as an `m` line, citing that `m` line's turn, with a different normalised value, corrects it. A
corrected `m` line is excluded from heads, from the index key text and from the rendered facts (it is shown only
as "replaced").
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import pickle
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from spinalmemory.project import entities as E
from spinalmemory.project import timeexpr as T
from spinalmemory.project.rendering import COST_HANDLE, annotated_text, fact_line, turn_line
from spinalmemory.project.vocab import VOCAB_HASH, is_single_valued
from spinalmemory.truth import SessionFile, TurnBlock

# Bump when the projection changes what it derives (it is part of the build fingerprint and the dump header).
PROJECTION_VERSION = "1"


@dataclass(frozen=True)
class Codecs:
    """Pinned non-text projection inputs: the reader tokenizer (rendered costs) and the encoder id (vector keys)."""

    count_tokens: Callable[[str], int]
    tokenizer_id: str
    encoder_id: str


def rowid_of(turn_id: str) -> int:
    """Injective-by-assertion FTS5 rowid: a function of the turn id alone, so insert order never matters."""
    return int.from_bytes(hashlib.sha256(turn_id.encode()).digest()[:8], "big") >> 1


def vector_key(encoder_id: str, text: str) -> str:
    return hashlib.sha256(f"{encoder_id}\n{text}".encode()).hexdigest()


def norm_value(value: str) -> str:
    s = E.norm_name(value)
    for art in ("a ", "an ", "the "):
        if s.startswith(art):
            s = s[len(art):]
    return s


def _iso(d: dt.date | None) -> str | None:
    return d.isoformat() if d else None


def _when_phrase(when: str) -> str | None:
    w = when.strip()
    if len(w) >= 2 and w[0] == '"' and w[-1] == '"':
        return w[1:-1].strip() or None
    return None


class Projection:
    def __init__(self, benchmark: str, codecs: Codecs, db_path: str | Path = ":memory:"):
        self.benchmark = benchmark
        self.codecs = codecs
        self.partitions = ("all", "user") if benchmark == "longmemeval" else ("all",)
        self.db_path = str(db_path)
        self._connect()
        self.files: dict[str, SessionFile] = {}
        self.blocks: dict[str, tuple] = {}                 # turn id -> (session, pos, date, block) as last projected
        # sessions and turns
        self.sessions: dict[str, dict] = {}
        self.turns: dict[str, dict] = {}
        # facts and their source turns
        self.facts: dict[str, dict] = {}
        self.fact_src: dict[str, tuple[str, ...]] = {}     # fid -> existing source turns, sorted
        self.src_index: dict[str, set[str]] = defaultdict(set)  # declared source turn -> fids
        # entities and aliases
        self.alias_pairs: dict[str, tuple[str, str]] = {}
        self.canon: dict[str, str] = {}
        self.known_names: Counter = Counter()
        self.speakers: Counter = Counter()
        self.name_index: dict[str, set[str]] = defaultdict(set)   # subject name -> fids
        self.value_index: dict[str, set[str]] = defaultdict(set)  # normalised value -> fids
        self.fact_ent: dict[str, tuple[str, str | None]] = {}
        # resolved time expressions
        self.timex: dict[str, list[dict]] = {}             # turn id or fact id -> expressions
        # attribute spines and correction links
        self.fact_lane: dict[str, tuple[str, str]] = {}    # fid -> its spine key (entity, attribute)
        self.lane_members: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.spines: dict[tuple[str, str], dict] = {}      # key -> {"head": fid | None, "members": [fid newest first]}
        self.corrected: dict[str, tuple[str, ...]] = {}    # m fid -> the h fids that correct it
        self.corrects: dict[str, tuple[str, ...]] = {}     # h fid -> the m fids it corrects
        # per-session count keys
        self.count_keys: dict[str, dict[tuple, list[str]]] = {}
        # per-turn index key text, rendered costs, vector keys
        self.keys: dict[str, str] = {}
        self.costs: dict[str, int] = {}
        self.fact_costs: dict[str, int] = {}               # tokens of the fact lines rendered under the turn
        self.vec_keys: dict[str, str] = {}
        self.rowids: dict[int, str] = {}

    # ------------------------------------------------------------------ persistence

    def _connect(self) -> None:
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        for p in self.partitions:
            self.db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS fts_{p} USING fts5(key, tokenize='porter unicode61')")
            self.db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vocab_{p} USING fts5vocab(fts_{p}, 'row')")
        self.db.commit()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("db")
        state["codecs"] = None  # functions are re-attached on load
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.db = None

    def save(self, path: Path) -> None:
        self.db.commit()
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(path)

    @staticmethod
    def load(path: Path, codecs: Codecs | None = None, readonly: bool = False) -> "Projection":
        """Load a saved projection; its FTS5 file is the `fts.sqlite` next to the pickle."""
        proj: Projection = pickle.loads(path.read_bytes())
        proj.codecs = codecs
        proj.db_path = str(path.parent / "fts.sqlite")
        if readonly:
            proj.db = sqlite3.connect(f"file:{proj.db_path}?mode=ro", uri=True, check_same_thread=False)
        else:
            proj._connect()
        return proj

    # ------------------------------------------------------------------ update

    def update(self, changes: dict[str, SessionFile | None]) -> dict:
        """Apply changed session files (None = deleted). Returns counts of what was re-derived."""
        d_keys: set[str] = set()       # turns whose index key / cost / FTS rows must be recomputed
        d_facts: set[str] = set()      # facts whose entity edges, lane and sources must be recomputed
        d_lanes: set[tuple[str, str]] = set()
        d_count: set[str] = set()      # sessions whose count keys must be recomputed
        d_names: set[str] = set()      # names that entered or left the entity name set
        alias_changed = False
        speakers_before = set(self.speakers)

        # Phase 1: which blocks changed (a block is a turn with its fact lines)
        removals, additions = [], []
        for sid in sorted(changes):
            new = copy.deepcopy(changes[sid])
            old_ids = [t for t, b in self.blocks.items() if b[0] == sid]
            new_blocks = {}
            if new is not None:
                for pos, blk in enumerate(new.turns):
                    if blk.id in new_blocks:
                        raise ValueError(f"duplicate turn id {blk.id} in {sid}")
                    new_blocks[blk.id] = (sid, pos, new.date, blk)
                self.files[sid] = new
                self.sessions[sid] = {"date": new.date, "raw_date": new.raw_date, "turns": len(new.turns)}
            else:
                self.files.pop(sid, None)
                self.sessions.pop(sid, None)
            for tid in sorted(set(old_ids) | set(new_blocks)):
                old_b, new_b = self.blocks.get(tid), new_blocks.get(tid)
                if old_b is not None and new_b is not None and _same_block(old_b, new_b):
                    continue
                if old_b is not None and old_b[0] == sid:
                    removals.append(tid)
                if new_b is not None:
                    additions.append((tid, new_b))
        # Phase 2: remove old blocks, then add new ones (so a turn may move between files in one update)
        for tid in removals:
            alias_changed |= self._remove_block(tid, d_keys, d_facts, d_lanes, d_count, d_names)
        for tid, b in additions:
            if tid in self.blocks:
                raise ValueError(f"turn id {tid} is in two sessions")
            alias_changed |= self._add_block(tid, b, d_keys, d_facts, d_count, d_names)
        touched_blocks = len(removals) + len(additions)

        # aliases: union-find over alias lines; names whose representative changed dirty the facts that use them
        if alias_changed or set(self.speakers) != speakers_before:
            old_canon = self.canon
            self.canon = E.canonical_map(sorted(self.alias_pairs.values()), set(self.speakers) | {"user", "assistant"})
            for n in set(old_canon) | set(self.canon):
                if old_canon.get(n, n) != self.canon.get(n, n):
                    d_facts |= self.name_index.get(n, set()) | self.value_index.get(n, set())
        for n in d_names:
            d_facts |= self.value_index.get(n, set())

        # fact edges, spine membership
        for fid in sorted(d_facts):
            f = self.facts.get(fid)
            if f is None:
                continue
            self.fact_src[fid] = tuple(sorted(t for t in f["sources"] if t in self.turns))
            subj = self._ent(f["subj_name"])
            obj = self._ent(f["val_name"]) if f["val_name"] in self.known_names else None
            self.fact_ent[fid] = (subj, obj)
            lane = (subj, f["attribute"]) if f["attribute"] != "alias" else None
            old_lane = self.fact_lane.get(fid)
            if old_lane != lane:
                if old_lane:
                    self.lane_members[old_lane].discard(fid)
                    d_lanes.add(old_lane)
                if lane:
                    self.lane_members[lane].add(fid)
                    self.fact_lane[fid] = lane
                else:
                    self.fact_lane.pop(fid, None)
                    self._set_links(fid, (), d_keys)  # out of every spine: no correction link
                    self._set_links(fid, (), d_keys, "corrects")
            if lane:
                d_lanes.add(lane)
            d_count.add(f["session"])

        for lane in sorted(d_lanes):
            self._derive_lane(lane, d_keys)
        for sid in sorted(d_count):
            self._derive_count_keys(sid)
        for tid in sorted(d_keys):
            self._derive_key(tid)
        self.db.commit()
        return {"blocks": touched_blocks, "keys": len(d_keys), "facts": len(d_facts), "spines": len(d_lanes), "count_sessions": len(d_count)}

    def _ent(self, name: str) -> str:
        return self.canon.get(name, name)

    def _remove_block(self, tid, d_keys, d_facts, d_lanes, d_count, d_names) -> bool:
        sid, _, _, blk = self.blocks.pop(tid)
        alias = False
        self.turns.pop(tid, None)
        self.timex.pop(tid, None)
        spk = E.norm_name(blk.speaker)
        self.speakers[spk] -= 1
        if self.speakers[spk] <= 0:
            del self.speakers[spk]
        d_keys.add(tid)
        d_count.add(sid)
        d_facts |= self.src_index.get(tid, set())  # facts citing this turn lose a source edge
        for fid in blk.fact_ids():
            f = self.facts.pop(fid)
            for other in self.corrected.pop(fid, ()) + self.corrects.pop(fid, ()):
                if other in self.facts:  # its spine is re-derived (d_lanes below), which re-links it
                    d_keys.update(self.facts[other]["sources"])
            self.timex.pop(fid, None)
            self.fact_src.pop(fid, None)
            self.fact_ent.pop(fid, None)
            for s in f["sources"]:
                self.src_index[s].discard(fid)
                if not self.src_index[s]:
                    del self.src_index[s]
                d_keys.add(s)
            self.name_index[f["subj_name"]].discard(fid)
            self.value_index[f["val_name"]].discard(fid)
            for n in f["names"]:
                self.known_names[n] -= 1
                if self.known_names[n] <= 0:
                    del self.known_names[n]
                    d_names.add(n)
            if fid in self.alias_pairs:
                del self.alias_pairs[fid]
                alias = True
            lane = self.fact_lane.pop(fid, None)
            if lane:
                self.lane_members[lane].discard(fid)
                d_lanes.add(lane)
        return alias

    def _add_block(self, tid, b, d_keys, d_facts, d_count, d_names) -> bool:
        sid, pos, date, blk = b
        self.blocks[tid] = b
        alias = False
        said_at = blk.said_at or date
        anchor = T.parse_iso_date(said_at)
        self.turns[tid] = {"session": sid, "pos": pos, "speaker": blk.speaker, "role": blk.role,
                           "said_at": said_at, "text": blk.text, "caption": blk.caption}
        self.timex[tid] = T.resolve(blk.text, anchor)
        self.speakers[E.norm_name(blk.speaker)] += 1
        d_keys.add(tid)
        d_count.add(sid)
        d_facts |= self.src_index.get(tid, set())  # facts citing this turn from elsewhere gain a source edge
        for fid, line in zip(blk.fact_ids(), blk.facts):
            subj = E.subject_name(line.subject, blk.speaker, blk.role)
            val = norm_value(line.value)
            names = [subj]
            if line.attribute == "alias" and val:
                self.alias_pairs[fid] = tuple(sorted((subj, val)))
                names.append(val)
                alias = True
            sources = tuple(dict.fromkeys((tid,) + tuple(line.also)))
            self.facts[fid] = {"home": tid, "session": sid, "tag": line.tag, "sentence": line.sentence,
                               "subject": line.subject, "attribute": line.attribute, "value": line.value,
                               "when": line.when, "sources": sources, "subj_name": subj, "val_name": val,
                               "names": names, "said_at": said_at, "order": (date, sid, pos)}
            self.timex[fid] = self._resolve_when(line.when, anchor)
            for s in sources:
                self.src_index[s].add(fid)
                d_keys.add(s)
            self.name_index[subj].add(fid)
            self.value_index[val].add(fid)
            for n in names:
                if self.known_names[n] == 0:
                    d_names.add(n)
                self.known_names[n] += 1
            d_facts.add(fid)
        return alias

    @staticmethod
    def _resolve_when(when: str, anchor: dt.date) -> list[dict]:
        iv = T.explicit_interval(when)
        if iv:
            return [{"kind": "explicit", "lo": iv[0], "hi": iv[1], "text": when, "start": 0, "end": len(when), "vague": False}]
        phrase = _when_phrase(when)
        return T.resolve(phrase, anchor)[:1] if phrase else []

    def valid_interval(self, fid: str) -> tuple[dt.date, dt.date] | None:
        """When the fact applies: its resolved, non-vague `when` (a human @date or the writer's time phrase).
        None = unknown; an unknown valid date is never filled in from the learned date."""
        ex = self.timex.get(fid) or []
        if ex and not ex[0]["vague"]:
            return ex[0]["lo"], ex[0]["hi"]
        return None

    def _spine_key(self, f: str):
        """Newest first (sorted in reverse): valid date if known else learned date, learned date, id hash."""
        v = self.valid_interval(f)
        learned = self.facts[f]["said_at"]
        return (v[0].isoformat() if v else learned[:10]), learned, rowid_of(f)

    def _set_links(self, fid: str, new: tuple[str, ...], d_keys: set[str], table: str = "corrected") -> None:
        """Set one fact's correction links; a change dirties the index keys and fact costs of every turn the facts
        on either side cite (their key text and rendered fact lines change)."""
        tab = getattr(self, table)
        old = tab.get(fid, ())
        if old == new:
            return
        for f in (fid,) + old + new:
            if f in self.facts:
                d_keys.update(self.facts[f]["sources"])
        if new:
            tab[fid] = new
        else:
            tab.pop(fid, None)

    def corrects_rule(self, h: str, m: str) -> bool:
        """The correction rule: h is a human line, m a machine line of the same spine, h cites m's turn, and their
        normalised values differ."""
        fh, fm = self.facts[h], self.facts[m]
        return (fh["tag"] == "h" and fm["tag"] == "m" and fm["home"] in fh["sources"]
                and norm_value(fh["value"]) != norm_value(fm["value"]))

    def _derive_lane(self, lane, d_keys: set[str] | None = None) -> None:
        d_keys = d_keys if d_keys is not None else set()
        members = self.lane_members.get(lane) or set()
        if not members:
            self.spines.pop(lane, None)
            self.lane_members.pop(lane, None)
            return
        order = sorted(members, key=self._spine_key, reverse=True)
        humans = [f for f in order if self.facts[f]["tag"] == "h"]
        machines = [f for f in order if self.facts[f]["tag"] == "m"]
        for m in machines:
            self._set_links(m, tuple(sorted(h for h in humans if self.corrects_rule(h, m))), d_keys)
        for h in humans:
            self._set_links(h, tuple(sorted(m for m in machines if self.corrects_rule(h, m))), d_keys, "corrects")
        head = None
        if is_single_valued(lane[1]):
            head = next((f for f in order if f not in self.corrected), None)
        self.spines[lane] = {"head": head, "members": order}

    def _derive_count_keys(self, sid: str) -> None:
        if sid not in self.sessions:
            self.count_keys.pop(sid, None)
            return
        groups: dict[tuple, list[str]] = defaultdict(list)
        for blk in self.files[sid].turns:
            for fid in blk.fact_ids():
                f = self.facts[fid]
                groups[(self.fact_ent[fid][0], f["attribute"], norm_value(f["value"]))].append(fid)
        self.count_keys[sid] = {k: sorted(v) for k, v in sorted(groups.items())}

    def turn_key(self, tid: str) -> str:
        """The index key text: "Speaker: text [image: caption] || fact sentences" (facts citing the turn, by fact id)."""
        t = self.turns[tid]
        key = f"{t['speaker']}: {t['text']}" + (f" [image: {t['caption']}]" if t["caption"] else "")
        sentences = [self.facts[f]["sentence"] for f in self.turn_facts(tid)]
        return key + (" || " + " ".join(sentences) if sentences else "")

    def turn_facts(self, tid: str) -> list[str]:
        """The facts citing a turn, by fact id, without corrected m lines (the correction rule)."""
        return [f for f in sorted(self.src_index.get(tid, ())) if f not in self.corrected]

    def valid_text(self, fid: str) -> str | None:
        v = self.valid_interval(fid)
        return T.calendar_form(*v) if v else None

    def fact_lines(self, tid: str) -> list[str]:
        """The fact lines rendered under a turn (and their cost): each fact citing the turn with its
        valid date when known; an h line that corrects an m line says so, with the value it replaced."""
        out = []
        for f in self.turn_facts(tid):
            replaced = [self.facts[m]["value"] for m in self.corrects.get(f, ())]
            out.append(fact_line(self.facts[f]["sentence"], self.valid_text(f), replaced))
        return out

    def rendered_text(self, tid: str, template: bool = True) -> str:
        t = self.turns[tid]
        return annotated_text(t["text"], self.timex.get(tid, []), T.parse_iso_date(t["said_at"]), template)

    def _derive_key(self, tid: str) -> None:
        rid = rowid_of(tid)
        for p in self.partitions:
            self.db.execute(f"DELETE FROM fts_{p} WHERE rowid = ?", (rid,))
        if tid not in self.turns:
            for table in (self.keys, self.costs, self.fact_costs, self.vec_keys):
                table.pop(tid, None)
            if self.rowids.get(rid) == tid:
                del self.rowids[rid]
            return
        if self.rowids.setdefault(rid, tid) != tid:
            raise AssertionError(f"FTS5 rowid collision between {tid} and {self.rowids[rid]}")
        key = self.turn_key(tid)
        t = self.turns[tid]
        self.keys[tid] = key
        self.vec_keys[tid] = vector_key(self.codecs.encoder_id, key)
        line = turn_line(COST_HANDLE, t["speaker"], self.rendered_text(tid), t["caption"])
        self.costs[tid] = self.codecs.count_tokens(line) + 1
        flines = self.fact_lines(tid)
        self.fact_costs[tid] = self.codecs.count_tokens("\n".join(flines)) + 1 if flines else 0
        for p in self.partitions:
            if p == "all" or t["role"] == p:
                self.db.execute(f"INSERT INTO fts_{p}(rowid, key) VALUES (?, ?)", (rid, key))

    # ------------------------------------------------------------------ views used by the query path

    def partition_ids(self, part: str) -> list[str]:
        """The turns of a partition in time order (session date, session id, position)."""
        ids = [t for t, r in self.turns.items() if part == "all" or r["role"] == part]
        return sorted(ids, key=self.time_key)

    def time_key(self, tid: str):
        t = self.turns[tid]
        return (self.sessions[t["session"]]["date"], t["session"], t["pos"])

    def session_turns(self, sid: str) -> list[str]:
        return [b.id for b in self.files[sid].turns]

    # ------------------------------------------------------------------ canonical dump

    def dump(self) -> dict:
        def iv(e):
            return [e["start"], e["end"], e["text"], e["kind"], _iso(e["lo"]), _iso(e["hi"]), e["vague"]]

        fts = {}
        for p in self.partitions:
            content = self.db.execute(f"SELECT rowid, key FROM fts_{p} ORDER BY rowid").fetchall()
            vocab = self.db.execute(f"SELECT term, doc, cnt FROM vocab_{p} ORDER BY term").fetchall()
            fts[p] = {"content": [list(r) for r in content], "vocab": [list(r) for r in vocab]}
        next_edges = []
        for sid in sorted(self.files):
            ids = self.session_turns(sid)
            next_edges += [[a, b] for a, b in zip(ids, ids[1:])]
        entities = defaultdict(list)
        for n in sorted(self.known_names):
            entities[self._ent(n)].append(n)
        return {
            "header": {
                "projection": PROJECTION_VERSION, "benchmark": self.benchmark, "resolver": T.RESOLVER_HASH,
                "vocabulary": VOCAB_HASH, "alias_rules": E.ALIAS_RULES_VERSION,
                "encoder": self.codecs.encoder_id, "tokenizer": self.codecs.tokenizer_id,
            },
            "sessions": [[s, v["date"], v["raw_date"], v["turns"]] for s, v in sorted(self.sessions.items())],
            "turns": [[t, r["session"], r["pos"], r["speaker"], r["role"], r["said_at"], r["text"], r["caption"]]
                      for t, r in sorted(self.turns.items())],
            "next": sorted(next_edges),
            "facts": [[f, r["home"], r["tag"], r["sentence"], r["subject"], r["attribute"], r["value"], r["when"]]
                      for f, r in sorted(self.facts.items())],
            "fact_src": sorted([f, t] for f, ts in self.fact_src.items() for t in ts),
            "fact_entity": sorted([f, s, o] for f, (s, o) in self.fact_ent.items()),
            "entities": [[e, ms] for e, ms in sorted(entities.items())],
            "timex": sorted([o] + iv(e) for o, es in self.timex.items() for e in es),
            "spines": [[s, a, v["head"], v["members"]] for (s, a), v in sorted(self.spines.items())],
            "corrected": sorted([m, list(hs)] for m, hs in self.corrected.items()),
            "corrects": sorted([h, list(ms)] for h, ms in self.corrects.items()),
            "fact_costs": sorted([t, c] for t, c in self.fact_costs.items()),
            "count_keys": [[sid, list(k), v] for sid, ks in sorted(self.count_keys.items()) for k, v in ks.items()],
            "keys": sorted([t, k] for t, k in self.keys.items()),
            "costs": sorted([t, c] for t, c in self.costs.items()),
            "vectors": sorted([t, k] for t, k in self.vec_keys.items()),
            "fts": fts,
        }

    def dump_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.dump(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _same_block(a: tuple, b: tuple) -> bool:
    """A block is unchanged if its session, position, session date and content are unchanged."""
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]


def full_build(benchmark: str, files: dict[str, SessionFile], codecs: Codecs, db_path: str | Path = ":memory:") -> Projection:
    proj = Projection(benchmark, codecs, db_path)
    proj.update(dict(files))
    return proj
