"""The temporal path: a question flag, question-date brackets and an instruction block (Notes are parsed in
render.py). A question takes this path when it carries its own asking date (input format) and the fixed rule
below flags its text; every other prompt keeps the standard instruction block.

- Flag: the kind of temporal question, from its text alone (no labels, no model), tested in this order:
    elapsed  "how many days/weeks/months/years ... ago/passed/between/before/after/since/until", "how long ago",
             "how long ... before/after/since/between/until/ago"
    order    "which ... first/earliest/most recently/latest/last" (not "last week/month/..."), "order",
             "first to last", "earliest"
    when     "When ...", "when did/was/were/had/have/is", "what date/day"
    ago      "N days/weeks/months/years ago", "last/past/this past/previous week/weekend/month/year", "yesterday"
    dated    a named fixed-date holiday before the question date (Valentine's Day, Christmas, Thanksgiving, ...)
  and the question's own past time phrases, resolved against the question date by the date resolver plus the
  holiday table (weekdays and anything on or after the question date are skipped).
- Memory block: every bracket gives the date and how long before the question date it is
  ("[= 2023-02-01 (Wednesday), 19 days ago]") instead of the template form anchored on the session date.
  Weekdays inside names or before an explicit date already have no bracket (the resolver's rule).
- Instruction block: the intro says what headers and brackets mean, a "Time in the question" line follows the
  question date, then a how-to line for the kind (elapsed: plus a conversion line for the unit asked), and the
  format Notes (at most 40 words) / Answer / Evidence. Counting questions of other kinds list the items in the
  Notes.
- Output: max_tokens 320 on flagged questions (config: temporal.max_tokens).
"""

from __future__ import annotations

import datetime as dt
import re

from spinalmemory.project import timeexpr as T

D = dt.timedelta

# ---------------------------------------------------------------- the flag (question text only)

UNIT = r"(?:days?|weeks?|months?|years?)"
NUM = (r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a couple of|couple of|a few|few|"
       r"several)")
ELAPSED = re.compile(
    rf"\bhow many {UNIT}\b[^?]*?\b(?:ago|pass|passed|elapsed|between|before|after|since|until|later|earlier)\b"
    r"|\bhow long ago\b"
    r"|\bhow long\b[^?]*?\b(?:before|after|since|between|until|ago)\b",
    re.I,
)
ORDER = re.compile(
    r"\bwhich\b[^?]*?\b(?:first|earliest|most recently|latest|last(?! (?:week|weekend|month|year|night|time)\b))\b"
    r"|\border\b|\bfirst to last\b|\bearliest\b",
    re.I,
)
WHEN = re.compile(r"^\W*when\b|\bwhen (?:did|was|were|had|have|is)\b|\bwhat (?:date|day)\b", re.I)
AGO = re.compile(
    rf"\b{NUM} {UNIT} ago\b|\b(?:last|past|this past|previous) (?:week|weekend|month|year)\b|\byesterday\b",
    re.I,
)
KINDS = ("elapsed", "order", "when", "ago", "dated")

# Fixed-date holidays (and Thanksgiving, computed) a question may name. Movable feasts are left out on purpose:
# conversations often give them dates that do not match the calendar.
HOLIDAYS = [
    (re.compile(r"\bvalentine'?s'? day\b", re.I), "Valentine's Day", (2, 14)),
    (re.compile(r"\bchristmas eve\b", re.I), "Christmas Eve", (12, 24)),
    (re.compile(r"\bchristmas(?: day)?\b(?! eve)", re.I), "Christmas Day", (12, 25)),
    (re.compile(r"\bnew year'?s eve\b", re.I), "New Year's Eve", (12, 31)),
    (re.compile(r"\bnew year'?s(?: day)?\b(?! eve)", re.I), "New Year's Day", (1, 1)),
    (re.compile(r"\bhalloween\b", re.I), "Halloween", (10, 31)),
    (re.compile(r"\bst\.? patrick'?s day\b|\bsaint patrick'?s day\b", re.I), "St. Patrick's Day", (3, 17)),
    (re.compile(r"\bindependence day\b|\bfourth of july\b|\b4th of july\b", re.I), "Independence Day", (7, 4)),
    (re.compile(r"\bthanksgiving\b", re.I), "Thanksgiving", "thanksgiving"),
]


def _thanksgiving(year: int) -> dt.date:
    d = dt.date(year, 11, 1)
    return d + D(days=(3 - d.weekday()) % 7 + 21)  # the fourth Thursday of November


def holiday_dates(text: str, qdate: dt.date) -> list[tuple[str, str, dt.date]]:
    """(phrase as asked, holiday name, its most recent date on or before the question date)."""
    out = []
    for pat, name, md in HOLIDAYS:
        m = pat.search(text)
        if not m:
            continue
        for y in (qdate.year, qdate.year - 1):
            d = _thanksgiving(y) if md == "thanksgiving" else T._safe_date(y, md[0], md[1])
            if d and d <= qdate:
                out.append((m.group(0), name, d))
                break
    return out


def temporal_kind(text: str, qdate: dt.date) -> str | None:
    """The temporal flag. "dated" is only a named holiday: an explicit date in a question ("in January", "the 7/22") is a
    lookup key with no time to reason about, and keeps the standard prompt."""
    if ELAPSED.search(text):
        return "elapsed"
    if ORDER.search(text):
        return "order"
    if WHEN.search(text):
        return "when"
    if AGO.search(text):
        return "ago"
    if any(d < qdate for _, _, d in holiday_dates(text, qdate)):
        return "dated"
    return None


def question_times(text: str, qdate: dt.date) -> list[dict]:
    """The question's own time phrases that lie before the question date, resolved against it. Weekday names are
    skipped ("on Monday" in a question is any Monday), and so is anything on or after the question date (plans).
    A month or day named without a year that the resolver places after the question date is moved back a year ("in
    December" asked in May means last December)."""
    out = []
    for e in T.resolve(text, qdate):
        if e["kind"].startswith("weekday") or e["kind"] == "weekend_bare":
            continue
        lo, hi = e["lo"], e["hi"]
        if lo > qdate and T.is_relative(e) and e["kind"] in ("day", "month") and re.search(T.MRE, e["text"].lower()):
            if e["kind"] == "month":
                lo, hi = T.month_iv(lo.year - 1, lo.month)
            else:
                lo = hi = T._safe_date(lo.year - 1, lo.month, lo.day)
        if lo is None or hi >= qdate:
            continue
        out.append({"text": e["text"], "lo": lo, "hi": hi, "name": None})
    for phrase, name, d in holiday_dates(text, qdate):
        if d < qdate:
            out.append({"text": phrase, "lo": d, "hi": d, "name": name})
    return out


# ---------------------------------------------------------------- brackets relative to the question date

def _days(n: int) -> str:
    return f"{n} day{'' if n == 1 else 's'}"


def days_before(lo: dt.date, hi: dt.date, qdate: dt.date) -> str:
    """How long before the question date an interval is: "19 days ago", "38-44 days ago", "in 3-4 days"."""
    a, b = (qdate - hi).days, (qdate - lo).days  # a <= b
    if a == b:
        return "the question date" if a == 0 else f"{_days(a)} ago" if a > 0 else f"in {_days(-a)}"
    if a >= 0:
        return f"{a}-{b} days ago"
    if b <= 0:
        return f"in {-b}-{-a} days"
    return f"from {_days(b)} ago to {_days(-a)} after the question date"


def date_note(lo: dt.date, hi: dt.date, qdate: dt.date) -> str:
    return f"{T.calendar_form(lo, hi)}, {days_before(lo, hi, qdate)}"


def annotated_text(text: str, exprs: list[dict], qdate: dt.date) -> str:
    """The turn text with each relative time phrase followed by "[= <date>, <days before the question date>]".
    Phrases with a year stay unannotated, as in the standard form."""
    out, last = [], 0
    for e in sorted(exprs, key=lambda e: e["start"]):
        if not T.is_relative(e):
            continue
        out.append(text[last:e["end"]])
        out.append(f" [= {date_note(e['lo'], e['hi'], qdate)}]")
        last = e["end"]
    out.append(text[last:])
    return "".join(out)


# ---------------------------------------------------------------- the instruction block

INTRO = ("Memory: turns from past conversations, sessions oldest first. [n] is a turn's handle. Each session header "
         "gives its date and how many days before the question date it was. A bracket after a time phrase gives the "
         "date it refers to and how long before the question date that is (\"19 days ago\").")

HOW = {
    "elapsed": ("This question asks how much time passed. Find the date of each event: the bracket date if its turn "
                "gives one, else that turn's session date. Count the days between them (or back from the question "
                "date)."),
    "order": ("This question asks about the order of events. Find the date of each event (the bracket date if its "
              "turn gives one, else that turn's session date) and compare the dates."),
    "when": ("This question asks when something happened. Find the turn that tells of it; its date is the bracket "
             "date if the turn gives one, else that turn's session date."),
    "ago": ("This question refers to a time before the question date. Use the session headers and brackets, which say "
            "how many days before the question date each turn is, to find the turns from that time."),
    "dated": ("This question names a time. Use the session headers and brackets to find the turns from that time; "
              "an event told in a session usually happened on or shortly before that session's date."),
}
NOTES_FORMAT = ("Reply in exactly this format:\n"
                "Notes: <at most 40 words: each event you need, with its handle and date, then the arithmetic or "
                "comparison>\n"
                "Answer: <the answer{unit}, as short as possible>\n"
                "Evidence: <the handles of the turns you used, e.g. [3] [7]>")
ENUMERATE_IN_NOTES = ("This question asks how many: list the items you counted in the Notes, then give the number on "
                      "the Answer line.")

# The unit the question asks for decides the conversion line (a generic line with an "e.g. 3 weeks" example makes
# the model answer "3 weeks" to a "how many days" question).
UNIT_ASKED = re.compile(r"\bhow many (day|week|month|year)s?\b", re.I)
UNIT_LINE = {
    "day": "Give the number of days.",
    "week": "Convert the days to weeks (7 days = 1 week) and round to the nearest whole week.",
    "month": "Convert to months (about 30 days = 1 month) and round to the nearest whole month.",
    "year": "Convert to years and round to the nearest whole year.",
    None: "Give the length of time in the unit that fits it best (days, weeks, months or years).",
}


def unit_asked(question: str) -> str | None:
    m = UNIT_ASKED.search(question)
    return m.group(1).lower() if m else None


def time_line(qtimes: list[dict], qdate: dt.date) -> str | None:
    """'Time in the question: "Valentine's day" (Valentine's Day) = 2023-02-14 (Tuesday), 16 days ago.'"""
    if not qtimes:
        return None
    parts = []
    for x in qtimes:
        label = f"\"{x['text']}\"" + (f" ({x['name']})" if x["name"] else "")
        parts.append(f"{label} = {date_note(x['lo'], x['hi'], qdate)}")
    return "Time in the question: " + "; ".join(parts) + "."


def render_prompt(question: str, memory: str, date_line: str, kind: str, tline: str | None, notes: str,
                  counting: bool) -> str:
    """The prompt for a flagged question; `notes` are render.intro_notes() for what the memory block shows."""
    intro = INTRO + notes
    dl = date_line + ("\n" + tline if tline else "")
    how = HOW[kind] + (" " + UNIT_LINE[unit_asked(question)] if kind == "elapsed" else "")
    unit = " and its unit" if kind == "elapsed" else ""
    outro = ("Answer the question from the memory. " + how + " First write brief notes, then the answer. "
             + NOTES_FORMAT.format(unit=unit))
    if counting and kind != "elapsed":
        outro += "\n" + ENUMERATE_IN_NOTES
    return f"Question: {question}\n\n{intro}\n\n{memory}\n\n{dl}\nQuestion: {question}\n\n{outro}"
