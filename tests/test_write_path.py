"""The write path: validation maps short ids back to turns and checks time spans, prose-only or example-id output
fails, commit keeps human lines, and the edit router re-projects fact edits with no writer call while a turn
edit re-ingests its session once."""

from __future__ import annotations

import asyncio
import copy

from _stubs import STUB_CODECS, DictCache, StubEncoder
from spinalmemory.config import load_config
from spinalmemory.project.projection import Projection, full_build
from spinalmemory.truth import FactLine, SessionFile, TurnBlock, read_store, write_session
from spinalmemory.write import build as B
from spinalmemory.write import edit as E
from spinalmemory.write.commit import commit
from spinalmemory.write.validate import Validated, validate
from spinalmemory.write.windows import make_window


def _session():
    return SessionFile("session_1", "2023-05-08T13:56", "", [
        TurnBlock("D1:1", "Priya", "speaker", "I joined a pottery class yesterday."),
        TurnBlock("D1:2", "Tom", "speaker", "That's great!"),
    ])


def _window():
    sess = _session()
    return sess, make_window(sess, "locomo")


def test_validate_maps_ids_and_checks_spans():
    _, w = _window()
    out = ('^t1\n- Priya joined a pottery class yesterday. | Priya | hobby | pottery | "yesterday"\n'
           '- Priya felt good. | Priya | Feeling Good | good | "last week"\n')
    v = validate(w, out)
    assert v.ok and list(v.lines) == ["D1:1"]
    first, second = v.lines["D1:1"]
    assert first.when == '"yesterday"' and first.attribute == "hobby"
    assert second.when == "-" and second.attribute == "other:feeling_good"  # not a literal span; coerced attribute
    assert v.stats["time_dropped"] == 1 and v.stats["attr_coerced"] == 1


def test_validate_cuts_time_phrases_the_window_never_said():
    _, w = _window()
    v = validate(w, "^t2\n- Tom was pleased last week. | Tom | feeling | pleased | -\n")
    assert v.lines["D1:2"][0].sentence == "Tom was pleased." and v.stats["time_phrases_cut"] == 1


def test_validate_rejects_prose_only_and_example_ids_and_caps_lines():
    _, w = _window()
    assert not validate(w, "Nothing in this session is worth remembering.").ok
    assert validate(w, "").ok  # an empty output is a valid "no facts"
    assert not validate(w, "^x1\n- Priya adopted a dog. | Priya | pet | dog | -\n").ok
    many = "^t1\n" + "".join(f"- Priya fact {k}. | Priya | event | e{k} | -\n" for k in range(5))
    v = validate(w, many, max_lines=3)
    assert len(v.lines["D1:1"]) == 3 and v.stats["over_cap"] == 2


def test_commit_keeps_human_lines():
    sess, w = _window()
    first = commit(sess, validate(w, "^t1\n- Priya joined. | Priya | hobby | pottery | -\n").lines)
    first.turns[0].facts.append(FactLine("h", "Human note.", "Priya", "hobby", "x"))
    again = commit(sess, validate(w, "^t2\n- Tom is glad. | Tom | feeling | glad | -\n").lines, previous=first)
    assert [f.tag for f in again.turns[0].facts] == ["h"]
    assert [f.tag for f in again.turns[1].facts] == ["m"]


def test_user_assistant_window_holds_user_turns_only():
    sess = SessionFile("s", "2023-05-08T13:56", "", [TurnBlock("s_1", "User", "user", "hi"),
                                                     TurnBlock("s_2", "Assistant", "assistant", "hello")])
    assert make_window(sess, "longmemeval").ids == (("t1", "s_1"),)


def test_edit_router(tmp_path, monkeypatch):
    calls = []

    async def stub_write(window, *a):
        calls.append(window.session_id)
        lines = {tid: [FactLine("m", f"Stub fact {short}.", "Priya", "hobby", "dance")] for short, tid in window.ids}
        return Validated(lines, {}, True), {"attempts": [{}]}, False

    monkeypatch.setattr(E, "write_window", stub_write)
    monkeypatch.setattr(E, "codecs_for", lambda cfg: STUB_CODECS)
    monkeypatch.setattr(E, "Encoder", StubEncoder)
    cache = DictCache()
    monkeypatch.setattr(B, "VectorCache", lambda: cache)
    cfg = load_config()
    store = tmp_path / "store"
    sess = SessionFile("session_1", "2023-03-01T10:00", "", [
        TurnBlock("D1:1", "Priya", "speaker", "I love dancing",
                  facts=[FactLine("h", "Priya teaches dance.", "Priya", "job", "dance teacher")]),
        TurnBlock("D1:2", "Tom", "speaker", "Me too", facts=[FactLine("m", "Tom likes dance.", "Tom", "likes", "dance")]),
    ])
    write_session(store, sess)
    (store / "index").mkdir()
    proj = full_build("locomo", read_store(store), STUB_CODECS, store / "index" / "fts.sqlite")
    B.materialise_vectors(proj, store / "index", StubEncoder(), cache)
    proj.save(store / "index" / "projection.pkl")

    line_edit = copy.deepcopy(sess)
    line_edit.turns[1].facts[0] = FactLine("h", "Tom loves dance.", "Tom", "likes", "dance")
    assert asyncio.run(E.apply_edit(store, line_edit, cfg))["writer_calls"] == 0 and calls == []

    turn_edit = copy.deepcopy(line_edit)
    turn_edit.turns[0].text = "I love salsa dancing"
    assert asyncio.run(E.apply_edit(store, turn_edit, cfg))["writer_calls"] == 1 and calls == ["session_1"]
    final = read_store(store)["session_1"]
    assert [f.tag for f in final.turns[0].facts] == ["m", "h"]  # new machine line, human line kept
    assert [f.tag for f in final.turns[1].facts] == ["m", "h"]
    incremental = Projection.load(store / "index" / "projection.pkl", STUB_CODECS)
    assert incremental.dump() == full_build("locomo", read_store(store), STUB_CODECS).dump()
