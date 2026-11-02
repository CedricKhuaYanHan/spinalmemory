"""Fact history: person spines order facts by valid date (else learned date) and keep the two dates apart, a
human line corrects a machine line only under the correction rule, the corrected value is what renders, and a
confirmed or duplicate fact renders once."""

from __future__ import annotations

import datetime as dt

from _stubs import STUB_CODECS
from spinalmemory.project.projection import full_build
from spinalmemory.query import render as R
from spinalmemory.truth import FactLine, SessionFile, TurnBlock

JUNE = dt.date(2023, 6, 1)


def _fl(sentence, subj, attr, val, when="-", tag="m", also=()):
    return FactLine(tag, sentence, subj, attr, val, when, tuple(also))


def _t(tid, spk, text, *facts):
    return TurnBlock(tid, spk, "speaker", text, facts=list(facts))


def _files():
    s1 = SessionFile("s1", "2023-03-01T10:00", "", [
        _t("D1:1", "Ana", "I work at the bakery now, and I love hiking",
           _fl("Ana works at the bakery.", "Ana", "job", "bakery"),
           _fl("Ana loves hiking.", "Ana", "hobby", "hiking")),
        _t("D1:2", "Omar", "Since 2019 I have lived in Leeds",
           _fl("Omar has lived in Leeds since 2019.", "Omar", "lives_in", "Leeds", "@2019")),
    ])
    s2 = SessionFile("s2", "2023-06-01T10:00", "", [
        _t("D2:1", "Ana", "I started at the library last week and took up pottery",
           _fl("Ana started at the library last week.", "Ana", "job", "library", '"last week"'),
           _fl("Ana took up pottery.", "Ana", "hobby", "pottery")),
        _t("D2:2", "Omar", "Ana also told me she is called Anabel",
           _fl("Ana is also called Anabel.", "Ana", "alias", "Anabel")),
    ])
    return {"s1": s1, "s2": s2}


class _Store:
    def __init__(self, proj):
        self.proj, self.has_roles = proj, False
        self.parts = {p: proj.partition_ids(p) for p in proj.partitions}


def _build(files):
    return full_build("locomo", files, STUB_CODECS)


def test_spines_have_heads_only_for_single_valued_attributes():
    p = _build(_files())
    assert set(p.spines) == {("ana", "job"), ("ana", "hobby"), ("omar", "lives_in")}  # no alias spine
    assert p.spines[("ana", "job")] == {"head": "D2:1#m01", "members": ["D2:1#m01", "D1:1#m01"]}
    assert p.spines[("ana", "hobby")] == {"head": None, "members": ["D2:1#m02", "D1:1#m02"]}


def test_valid_and_learned_dates_are_separate_and_unknown_stays_unknown():
    p = _build(_files())
    assert p.valid_interval("D1:1#m01") is None                         # no time phrase: unknown, with no said-at fallback
    assert p.facts["D1:1#m01"]["said_at"] == "2023-03-01T10:00"         # learned
    assert p.valid_interval("D1:2#m01") == (dt.date(2019, 1, 1), dt.date(2019, 12, 31))
    assert p.valid_text("D2:1#m01").startswith("2023-05-2")              # "last week" from 1 June
    files = _files()
    files["s2"].turns[1].facts.append(_fl("Omar lived in York in 2018.", "Omar", "lives_in", "York", "@2018"))
    files["s2"].turns[1].facts.append(_fl("Omar lives in Hull.", "Omar", "lives_in", "Hull"))
    # newest first by valid date if known, else learned date: the 2019 fact is older than anything said in 2023
    assert _build(files).spines[("omar", "lives_in")]["members"] == ["D2:2#m03", "D1:2#m01", "D2:2#m02"]


def test_correction_rule():
    files = _files()
    # same turn, same (entity, attribute), different value: the human line corrects the machine line
    files["s2"].turns[0].facts.append(_fl("Ana started at the city library.", "Ana", "job", "city library", tag="h"))
    # the same normalised value only confirms: no link
    files["s1"].turns[0].facts.append(_fl("Ana works at The Bakery.", "Ana", "job", "The Bakery", tag="h"))
    p = _build(files)
    assert p.corrected == {"D2:1#m01": ("D2:1#h01",)} and p.corrects == {"D2:1#h01": ("D2:1#m01",)}
    assert p.spines[("ana", "job")]["head"] != "D2:1#m01"
    assert "Ana started at the library last week." not in p.keys["D2:1"]  # out of the index key text
    assert "Ana started at the city library." in p.keys["D2:1"]
    # a human line on another turn corrects only when it cites the machine line's turn (also:)
    files = _files()
    files["s2"].turns[1].facts.append(_fl("Ana works at the museum.", "Ana", "job", "museum", tag="h"))
    assert _build(files).corrected == {}
    files["s2"].turns[1].facts[-1] = _fl("Ana works at the museum.", "Ana", "job", "museum", tag="h", also=("D2:1",))
    assert _build(files).corrected == {"D2:1#m01": ("D2:2#h01",)}


def test_corrected_value_renders_in_place_of_the_machine_value():
    files = _files()
    files["s2"].turns[0].facts.append(_fl("Ana started at the city library.", "Ana", "job", "city library", tag="h"))
    mem, _, info = R.render_memory(_Store(_build(files)), ["D1:1", "D2:1"], JUNE, True, False, True,
                                   [("ana", "job"), ("ana", "hobby")], 8)
    hist = mem.split("\n\n")[0]
    assert hist.startswith(R.HISTORY_HEAD)
    assert ("- Ana, job: city library [2] (current, corrected; replaced: library, said 2023-06-01); "
            "bakery [1] (earlier, said 2023-03-01)") in hist
    assert "- Ana, hobby: pottery [2] (said 2023-06-01); hiking [1] (said 2023-03-01)" in hist
    assert "Ana started at the library last week." not in mem
    assert "  - Ana started at the city library. (corrected; replaced: library)" in mem
    assert info == {"history_lines": 2, "fact_lines": 4, "corrected_shown": 1, "history_current": 1,
                    "history_corrected": 1}


def test_confirmed_or_duplicate_fact_renders_once():
    files = _files()
    files["s1"].turns[0].facts.append(_fl("Ana works at The Bakery.", "Ana", "job", "The Bakery", tag="h"))
    files["s2"].turns[0].facts.append(_fl("Ana took up pottery.", "Ana", "hobby", "pottery"))  # duplicate line
    p = _build(files)
    assert p.corrected == {}  # a confirmation is not a correction
    mem, _, info = R.render_memory(_Store(p), ["D1:1", "D2:1"], JUNE, True, False, True,
                                   [("ana", "job"), ("ana", "hobby")], 8)
    assert "  - Ana works at The Bakery." in mem and "Ana works at the bakery." not in mem
    assert mem.count("  - Ana took up pottery.") == 1
    job = next(ln for ln in mem.splitlines() if ln.startswith("- Ana, job:"))
    assert "The Bakery [1]" in job and "bakery [1]" not in job.replace("The Bakery [1]", "")
    assert info["fact_lines"] == 4 and info["corrected_shown"] == 0
