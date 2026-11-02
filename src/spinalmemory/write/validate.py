"""Validate: parse and check one writer output, in code.

Per line: the turn id must be one given in this call; five fields; non-empty sentence, subject and value;
the attribute in the closed vocabulary or `other:<snake_case>` (a free-form snake_case attribute is kept as
`other:<it>` and counted); the time phrase a literal span of the turn (else it is dropped to `-` and counted);
at most `max_lines` lines per turn. Short ids are mapped back to turn ids.

The sentence's own time phrases are checked too. A phrase the date resolver finds in the sentence that occurs
nowhere in the window the writer saw ("just launched" rewritten as "launched ... last week") is cut from the
sentence, with a preposition that only introduced it ("since last year") or a hedge on it ("about two weeks
ago"), and counted. A phrase counts as in the window when its core does: the phrase without leading
prepositions, articles and hedges ("on Sunday" -> "sunday", "a weekend" -> "weekend", "in 1908" -> "1908"),
compared word by word with case, markup, punctuation, ordinal suffixes, number words and weekday or month
abbreviations set aside. It is code over the cached output, so it never needs a writer call, and it never
changes whether a window passes.

The window fails (and is re-ingested once at write time) if the output uses the prompt example's ids, if it has
text but no parseable block, or if more than 20% of its lines are invalid.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from spinalmemory.project.timeexpr import resolve
from spinalmemory.project.vocab import valid_attribute
from spinalmemory.truth import FactLine, clean_field
from spinalmemory.write.windows import Window

# Bump when a rule changes which lines are kept or when a window is re-ingested (it is part of the build fingerprint).
VALIDATOR_VERSION = "1"
EXAMPLE_IDS = {"x1", "x2"}
HEADER = re.compile(r"^\^\s*([A-Za-z]+\d+)\b")
LINE = re.compile(r"^\s*[-*]\s+(.+)$")
MAX_INVALID = 0.2

# The writer never sees the session date, so only the phrase spans matter here; a fixed anchor keeps the check a
# function of the text alone (31 January of a leap year, so "the 31st" and "February 29" are valid dates).
PHRASE_ANCHOR = dt.date(2000, 1, 31)
LEAD_PREPOSITION = re.compile(r"(?:^|\s)(?:since|from|until|till|by|in|on|at|during|around|over|for|before|after|as of)\s*$", re.I)
HEDGE = re.compile(r"(?:^|\s)(?:about|almost|nearly|roughly|approximately|around|some|just|only)\s*$", re.I)
# Words that only introduce a time phrase; the words that carry the time ("last", "next", "this", "ago") stay.
LEAD_WORDS = re.compile(r"^(?:(?:on|in|at|by|from|since|until|till|during|around|over|for|of|back|as|the|a|an|my|"
                        r"our|your|his|her|their|its|about|almost|nearly|roughly|approximately|some|just|only) )+")
ABBREVIATIONS = {"mon": "monday", "tue": "tuesday", "tues": "tuesday", "wed": "wednesday", "weds": "wednesday",
                 "thu": "thursday", "thur": "thursday", "thurs": "thursday", "fri": "friday", "sat": "saturday",
                 "sun": "sunday", "jan": "january", "feb": "february", "mar": "march", "apr": "april", "jun": "june",
                 "jul": "july", "aug": "august", "sep": "september", "sept": "september", "oct": "october",
                 "nov": "november", "dec": "december"}
NUMBER_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


@dataclass
class Validated:
    lines: dict[str, list[FactLine]] = field(default_factory=dict)   # turn id -> machine lines
    stats: dict = field(default_factory=dict)
    ok: bool = True


def _attribute(raw: str) -> tuple[str, bool]:
    a = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if valid_attribute(a):
        return a, False
    a = re.sub(r"[^a-z0-9_]", "", a.removeprefix("other:"))[:40] or "unspecified"
    return f"other:{a}", True


def _when(raw: str, text: str) -> tuple[str, bool]:
    w = raw.strip()
    if w in ("", "-", "none", "None", "n/a"):
        return "-", False
    phrase = w.strip('"“”\'').strip()
    if phrase and phrase.lower() in text.lower():
        return f'"{clean_field(phrase)}"', False
    return "-", True


def _words(text: str) -> str:
    """Lower-case words, space-separated: markup, punctuation, hyphens and ordinal suffixes dropped, number words
    as digits, weekday and month abbreviations spelled out."""
    t = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", text.lower().replace("’", "'"))
    words = (w.strip("'") for w in re.sub(r"[^a-z0-9']+", " ", t).split())
    return " ".join(ABBREVIATIONS.get(w, NUMBER_WORDS.get(w, w)) for w in words if w)


def _core(phrase: str) -> str:
    """The phrase's words without the leading ones that only introduce it ("on sunday" -> "sunday")."""
    return LEAD_WORDS.sub("", _words(phrase) + " ").strip()


def unsourced_time_phrases(sentence: str, source: str) -> list[dict]:
    """Time expressions of the sentence (the resolver's spans) whose core occurs nowhere in `source`, word for word."""
    src = f" {_words(source)} "
    out = []
    for e in resolve(sentence, PHRASE_ANCHOR):
        core = _core(e["text"])
        if core and f" {core} " not in src:
            out.append(e)
    return out


def cut_phrases(sentence: str, exprs: list[dict]) -> str:
    """The sentence without the given spans, a hedge on one ("about two weeks ago"), a preposition that only
    introduced one (the phrase ends a clause: "since last year."), a possessive on one ("last year's"), and the
    spacing they leave."""
    out, last = [], 0
    for e in sorted(exprs, key=lambda e: e["start"]):
        head, end = sentence[last:e["start"]], e["end"]
        possessive = re.match(r"['’]s\b", sentence[end:])
        if possessive:
            end += possessive.end()
        if h := HEDGE.search(head):
            head = head[: h.start()] + " "
        m = LEAD_PREPOSITION.search(head)
        if m and not possessive and re.match(r"\s*(?:[.,;:!?)]|$)", sentence[end:]):
            head = head[: m.start()]
        out.append(head)
        last = end
    out.append(sentence[last:])
    s = re.sub(r"\s{2,}", " ", "".join(out))
    s = re.sub(r"\s+([.,;:!?)])", r"\1", s)
    s = re.sub(r",(?=[.!?;:]|$)", "", s.strip())
    s = re.sub(r"^[\s,;:]+", "", s)
    return s[:1].upper() + s[1:]


def validate(window: Window, output: str, max_lines: int = 3) -> Validated:
    short_to_turn = dict(window.ids)
    seen_text = "\n".join(window.texts.values())
    st = {"lines": 0, "kept": 0, "invalid": 0, "unknown_id": 0, "example_ids": 0, "over_cap": 0,
          "attr_coerced": 0, "time_dropped": 0, "blocks": 0,
          "time_phrases_cut": 0, "time_lines_changed": 0, "time_lines_dropped": 0}
    out: dict[str, list[FactLine]] = {}
    cur = None
    for raw in output.splitlines():
        s = raw.strip()
        m = HEADER.match(s)
        if m:
            sid = m.group(1).lower()
            st["blocks"] += 1
            if sid in EXAMPLE_IDS and sid not in short_to_turn:
                st["example_ids"] += 1
            cur = sid if sid in short_to_turn else None
            if cur is None:
                st["unknown_id"] += 1
            continue
        lm = LINE.match(s)
        if not lm:
            continue
        st["lines"] += 1
        parts = [p.strip() for p in lm.group(1).split("|")]
        if cur is None or len(parts) != 5 or not parts[0] or not parts[1] or not parts[3]:
            st["invalid"] += 1
            continue
        sentence, subject, attr, value, when = parts
        attr, coerced = _attribute(attr)
        when, dropped = _when(when, window.texts[cur])
        st["attr_coerced"] += coerced
        st["time_dropped"] += dropped
        tid = short_to_turn[cur]
        if len(out.get(tid, [])) >= max_lines:
            st["over_cap"] += 1
            continue
        stray = unsourced_time_phrases(sentence, seen_text)
        if stray:
            sentence = cut_phrases(sentence, stray)
            st["time_phrases_cut"] += len(stray)
            st["time_lines_changed"] += 1
            if not re.search(r"\w", sentence):  # nothing but the phrase: no fact left to state
                st["time_lines_dropped"] += 1
                continue
        out.setdefault(tid, []).append(FactLine("m", clean_field(sentence), clean_field(subject), attr, clean_field(value), when))
        st["kept"] += 1
    ok = st["example_ids"] == 0 and not (output.strip() and st["blocks"] == 0)  # text but no ^ block: fails
    if st["lines"] and st["invalid"] / st["lines"] > MAX_INVALID:
        ok = False
    return Validated(out, st, ok)
