"""The Markdown truth files round-trip exactly (in memory and on disk), turn text may hold Markdown, and time
phrases resolve to dates shown in both a calendar and a session-anchored form."""

from __future__ import annotations

import datetime as dt
import random

from _stubs import Store
from spinalmemory.project.timeexpr import annotate, resolve
from spinalmemory.truth import FactLine, SessionFile, TurnBlock, parse_session, read_store, write_session
from spinalmemory.write.append import iso_session_date

ANCHOR = dt.date(2023, 5, 8)


def _annotations(text: str) -> list[str | None]:
    return [annotate(e, ANCHOR) for e in resolve(text, ANCHOR)]


def test_random_stores_round_trip_in_memory_and_on_disk(tmp_path):
    for seed in range(10):
        store = Store(random.Random(seed), "locomo" if seed % 2 else "longmemeval")
        for sess in store.files.values():
            assert parse_session(sess.render()) == sess
        root = tmp_path / f"s{seed}"
        for sess in store.files.values():
            write_session(root, sess)
        assert read_store(root) == store.files


def test_text_with_markdown_inside_round_trips():
    text = "### not a header\n- not a fact | x\n> quoted\n\n  indented\n"
    sess = SessionFile("s1", "2023-05-08T13:56", "raw", [TurnBlock("s1_1", "User", "user", text)])
    assert parse_session(sess.render()).turns[0].text == text


def test_fact_lines_keep_dates_sources_and_drop_separators():
    line = FactLine("h", "Priya moved | again.", "Priya", "lives_in", "Leeds", "@2022-09", ("D1:3", "D2:1"))
    rendered = line.render()
    assert rendered == "- h | Priya moved / again. | Priya | lives_in | Leeds | @2022-09 | also: D1:3 D2:1"
    sess = SessionFile("s", "2023-05-08T13:56", "", [TurnBlock("D1:1", "Priya", "speaker", "hi", facts=[line])])
    back = parse_session(sess.render()).turns[0].facts[0]
    assert back.when == "@2022-09" and back.also == ("D1:3", "D2:1") and back.sentence == "Priya moved / again."
    assert sess.turns[0].fact_ids() == ["D1:1#h01"]


def test_dates_in_both_forms():
    assert _annotations("I went yesterday") == ["[= 2023-05-07 (Sunday); the day before 8 May 2023]"]
    assert _annotations("last week") == ["[= 2023-05-01 to 2023-05-07; the week before 8 May 2023]"]
    assert _annotations("last Friday") == ["[= 2023-05-05 (Friday); the Friday before 8 May 2023]"]
    assert _annotations("on 7 May 2023") == [None]  # an explicit date needs no annotation
    assert annotate(resolve("yesterday", ANCHOR)[0], ANCHOR, template=False) == "[= 2023-05-07 (Sunday)]"


def test_session_dates_in_both_input_formats():
    assert iso_session_date("1:56 pm on 8 May, 2023") == "2023-05-08T13:56"
    assert iso_session_date("2023/05/20 (Sat) 02:21") == "2023-05-20T02:21"
