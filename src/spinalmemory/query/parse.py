"""Parse the question by code: named persons (alias lookup), the role partition, the question date.

Branches are on the input format only:
- Named persons: speakers (turns with role "speaker") whose name or an alias of it occurs in the question.
  They form the third RRF list and are the matching persons of the fact-history check. A history whose turns
  have user/assistant roles has no named speakers.
- Partition: when the history has user/assistant roles, questions search the user turns unless the fixed
  "refers to the assistant" regex fires, in which case they search all turns. Histories without roles search
  all turns.
- Question date: the question's own date (`question["question_date"]`) when it carries one, else the last
  session date.
- Counting flag: "how many" / "number of" questions get the enumerate-then-answer line, only when they count
  distinct items. "How many weeks ago" or "the total number of days" ask for a quantity in a unit, and the line
  would make the model count the wrong things.
- Temporal kind (query/temporal.py): only for a question that carries its own date, by a fixed text rule.
  Advice flag (query/advice.py): a fixed text rule, for every question. Temporal takes precedence.
- Retrieval text: the question without the fixed suffix " Use DATE of CONVERSATION to answer with an approximate
  date." when the question ends with it (a property of the text); the prompt keeps the question as asked.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from spinalmemory.project.entities import norm_name
from spinalmemory.write.append import iso_session_date

ASSISTANT_REF = re.compile(
    r"\byou\b[^?.]*\b(said|told|mentioned|recommended|suggested|gave|provided|listed|shared|explained|wrote|made|came up with|talked about|discussed|described|named|offered|helped)\b"
    r"|\bremind me\b|\bwe (talked|discussed|chatted|spoke)\b|\bour (previous|last|earlier|past) (chat|conversation|discussion|game|session)\b",
    re.I,
)

STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being do does did done
i me my mine we our ours you your yours he him his she her hers it its they them their theirs this that these those
what which who whom whose when where why how can could would should will shall may might must have has had having
not no so than too very just about into over after before up down out off again then once there here all any both
each few more most other some such only own same s t don now also ever much many often long kind type sort thing things
get got go went going take took make made tell told say said mention mentioned remember recall ago since last
day days week weeks month months year years time times date first second third recently currently current
usually still yet already lot lots""".split())
TEMPORAL_Q = set("when date time year month day week weekend".split())
# A fixed instruction some question sets append to date questions; it is cut from the retrieval text only.
DATE_QUESTION_SUFFIX = " Use DATE of CONVERSATION to answer with an approximate date."
COUNTING = re.compile(r"\bhow many\b|\b(?:total )?number of\b", re.I)
# A unit of quantity right after "how many" / "number of" (with an optional modifier): the question asks for an
# amount, so no distinct items are listed. Units of time, distance, weight, money, volume and similar; "times",
# "people", "weekends" or "episodes" are left out, since listing them is how they are counted.
QUANTITY = re.compile(
    r"\b(?:how many|(?:total )?number of)\s+(?:more\s+|total\s+|full\s+|additional\s+|extra\s+|fewer\s+|less\s+|in\s+)?"
    r"(?:seconds?|minutes?|hours?|days?|nights?|weeks?|months?|years?|decades?|miles?|kilometers?|kilometres?|km|"
    r"meters?|metres?|feet|foot|inches|pounds?|lbs|kilograms?|kgs?|grams?|ounces?|pages?|dollars?|euros?|cents?|"
    r"calories|percent|liters?|litres?|gallons?|cups?|servings?|points|steps)\b",
    re.I,
)
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SUFFIXES = ("ings", "ing", "edly", "ed", "ies", "es", "s", "ly")


def stem(w: str) -> str:
    """A small deterministic suffix stripper (PyStemmer is not in the pinned environment)."""
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def words(text: str) -> list[str]:
    return [w.split("'")[0] for w in WORD.findall(text.lower())]


def content_stems(question: str, hubs=()) -> list[str]:
    hub = {h.lower() for h in hubs}
    keep = [w for w in words(question) if w not in STOP and w not in TEMPORAL_Q and w not in hub and len(w) > 1 and not w.isdigit()]
    return list(dict.fromkeys(stem(w) for w in keep))


def coverage(qstems: list[str], text: str) -> float:
    if not qstems:
        return 0.0
    ts = {stem(w) for w in words(text)}
    return sum(1 for w in qstems if w in ts) / len(qstems)


def fts_query(question: str) -> str | None:
    """FTS5 MATCH expression: the question's non-stop words, OR-ed (BM25 does the weighting)."""
    toks = [w for w in dict.fromkeys(re.findall(r"[a-z0-9]+", question.lower())) if w not in STOP and len(w) > 1]
    return " OR ".join(f'"{t}"' for t in toks) or None


def counts_items(text: str) -> bool:
    """The counting flag: "how many" / "number of" asking for a count of distinct items. Each occurrence is read on
    its own, so "How many days ... and how many museums ...?" still counts the museums."""
    return len(COUNTING.findall(text)) > len(QUANTITY.findall(text))


@dataclass
class Parsed:
    text: str                  # the question as asked (rendered in the prompt)
    retrieval_text: str        # what search and rerank use
    partition: str
    named: list[str]           # display names of the named speakers
    question_date: dt.date
    question_date_iso: str
    counting: bool = False     # render the enumerate-then-answer line
    temporal: str | None = None  # temporal kind (questions with their own date): elapsed, order, when, ago, dated
    advice: bool = False       # the question asks for a recommendation or advice
    own_date: bool = False     # the question carries its own asking date


def parse_question(question: dict, store, count_items_only: bool = True) -> Parsed:
    from spinalmemory.query.advice import is_advice
    from spinalmemory.query.temporal import temporal_kind

    text = question["question"]
    counting = counts_items(text) if count_items_only else bool(COUNTING.search(text))
    roles = store.has_roles                                                 # input format: user/assistant turns
    part = "user" if roles and "user" in store.parts and not ASSISTANT_REF.search(text) else "all"
    own_date = bool(question.get("question_date"))                          # input format: the question's date
    qd = iso_session_date(question["question_date"]) if own_date else store.last_date
    qdate = dt.date.fromisoformat(qd[:10])
    named = []
    for display, aliases in store.speaker_aliases.items():                 # input: persons named in the text
        if any(re.search(rf"\b{re.escape(a)}\b", text, flags=re.I) for a in aliases):
            named.append(display)
    kind = temporal_kind(text, qdate) if own_date else None
    return Parsed(text, text.removesuffix(DATE_QUESTION_SUFFIX), part, sorted(named), qdate, qd, counting, kind,
                  is_advice(text), own_date)


def speaker_aliases(proj) -> dict[str, list[str]]:
    """Display speaker -> the names that refer to them (their own name plus aliases), by code."""
    out: dict[str, list[str]] = {}
    for t in proj.turns.values():
        if t["role"] != "speaker":
            continue
        out.setdefault(t["speaker"], [])
    for display in out:
        me = norm_name(display)
        names = {me} | {n for n, rep in proj.canon.items() if rep == proj.canon.get(me, me)}
        out[display] = sorted(names)
    return dict(sorted(out.items()))
