"""Deterministic time-expression resolver (code, pinned; no LLM, no dateparser).

`resolve(text, anchor)` finds every time expression in a text with its character span and resolves it to an
inclusive date interval [lo, hi] against the anchor (the session date). Two rules worth knowing:
  - "the day after tomorrow" is anchor + 2 (not its "tomorrow": anchor + 1);
  - a bare weekday gets no date when it is part of a name ("Ash Wednesday": a capitalised word before it,
    mid-sentence) or when an explicit calendar date follows it in the same sentence ("the Sunday market
    on January 2nd"). Its span stays claimed, so no other rule matches inside it.
`annotate()` below adds the two rendered forms shown to the answering model: a calendar form ("2023-05-07
(Sunday)") and a template form anchored on the session date ("the day before 8 May 2023"). The resolver is a
projection input: its source hash goes into the canonical dump header.

Expression classes:
  precise   day, weekday, week, weekend, month, season, year, N-units-ago, duration-start ("for 3 years")
  vague     "a few days ago", "the other day" (interval [d-6, d-1]); resolved but flagged
Not expressions: "recently", "lately", "these days", "now", "N days later" (relative to another event).
"""

from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import re
from pathlib import Path

D = dt.timedelta
WD = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MON = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                   "september", "october", "november", "december"], start=1)}
MON_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
MON_ALL = {**MON, **MON_ABBR}
MRE = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "a couple of": 2, "couple of": 2, "a couple": 2,
        "couple": 2, "a few": 3, "few": 3, "several": 3}
NRE = r"(\d+|a couple of|couple of|a couple|a few|few|several|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)"
SEASON = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2)}


def _n(s):
    s = s.lower()
    return int(s) if s.isdigit() else NUMW.get(s)


def month_iv(y, m):
    return dt.date(y, m, 1), dt.date(y, m, calendar.monthrange(y, m)[1])


def add_months(d, k):
    m = d.month - 1 + k
    return d.year + m // 12, m % 12 + 1


def season_iv(y, s):
    a, b = SEASON[s]
    if s == "winter":
        return dt.date(y, 12, 1), dt.date(y + 1, 2, 28)
    return dt.date(y, a, 1), dt.date(y, b, calendar.monthrange(y, b)[1])


def _safe_date(y, m, d):
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def _infer_year_day(m, d, anchor, future=False):
    """Day-month without a year: the anchor's year, or the previous year if that lies more than 7 days ahead;
    up to 183 days ahead when the preceding words announce a plan ("I plan to return on July 20")."""
    x = _safe_date(anchor.year, m, d)
    if x is None:
        return None
    if x > anchor + D(days=183 if future else 7):
        x = _safe_date(anchor.year - 1, m, d) or x
    return x


def _infer_year_month(m, anchor, mode):
    if mode == "last":
        return anchor.year if m < anchor.month else anchor.year - 1
    if mode == "next":
        return anchor.year if m > anchor.month else anchor.year + 1
    return anchor.year  # "in May", "this May": the anchor year


# A sentence ends at . ! ? (not after a common abbreviation such as "St.") or a line break.
SENT_END = re.compile(r"(?<!\bSt)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bMt)(?<!\bvs)(?<!\betc)[.!?](?:\s|$)|\n")
SAME_SENTENCE_CHARS = 160
WORD_BEFORE = re.compile(r"([A-Za-z][\w'.-]*)\W*$")


def weekday_in_name(text: str, start: int) -> bool:
    """A bare weekday preceded by a capitalised word mid-sentence is part of a name ("Ash Wednesday",
    "Maundy Thursday", "Super Bowl Sunday"); at a sentence start ("Last Sunday", "I") it is not."""
    before = text[:start]
    m = WORD_BEFORE.search(before)
    if m and m.group(1)[0].isupper() and m.group(1) != "I":
        rest = before[: m.start(1)].rstrip()
        return bool(rest) and not re.search(r"[.!?:\"(]$", rest)
    return False


def explicit_date_follows(text: str, end: int, found: list[dict]) -> bool:
    """An explicit calendar date (a day expression with a digit) later in the same sentence, within 160
    characters, says when; the weekday must not add a second, guessed date."""
    tail = text[end:]
    cut = SENT_END.search(tail)
    stop = end + len((tail[: cut.start()] if cut else tail)[:SAME_SENTENCE_CHARS])
    return any(x["kind"] == "day" and end <= x["start"] < stop and re.search(r"\d", x["text"]) for x in found)


def resolve(text: str, anchor: dt.date) -> list[dict]:
    """All time expressions in `text`, resolved against `anchor`. Non-overlapping, sorted by position."""
    t = text.lower()
    taken = [False] * (len(t) + 1)
    out = []

    def free(a, b):
        return not any(taken[a:b])

    def add(m, kind, lo, hi, vague=False, group=0):
        a, b = m.span(group)
        if lo is None or hi is None or not free(a, b):
            return
        for i in range(a, b):
            taken[i] = True
        out.append({"kind": kind, "lo": lo, "hi": hi, "text": text[a:b], "start": a, "end": b, "vague": vague})

    def claim(m):
        """Take the span without emitting an expression (a weekday that names no date)."""
        a, b = m.span()
        if free(a, b):
            for i in range(a, b):
                taken[i] = True

    def it(p):
        return re.finditer(p, t)

    # ---- absolute dates, longest forms first
    for m in it(rf"\b{MRE} (\d{{1,2}})(?:st|nd|rd|th)?,? ((?:19|20)\d\d)\b"):
        x = _safe_date(int(m[3]), MON_ALL[m[1]], int(m[2]))
        add(m, "day", x, x)
    for m in it(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? (?:of )?{MRE},? ((?:19|20)\d\d)\b"):
        x = _safe_date(int(m[3]), MON_ALL[m[2]], int(m[1]))
        add(m, "day", x, x)
    for m in it(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)?\d\d)\b"):
        y = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
        x = _safe_date(y, int(m[1]), int(m[2]))
        add(m, "day", x, x)
    def fut(m):
        return bool(re.search(r"\b(will|'ll|plan|planning|going to|gonna|next|upcoming|scheduled|until|till|return)\b", t[max(0, m.start() - 40):m.start()]))

    for m in it(rf"\b{MRE} (\d{{1,2}})(?:st|nd|rd|th)?\b(?!:)"):
        if m[1] == "may" and not text[m.start():m.start() + 1].isupper():
            continue
        if int(m[2]) > 31:
            continue
        x = _infer_year_day(MON_ALL[m[1]], int(m[2]), anchor, fut(m))
        add(m, "day", x, x)
    for m in it(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? (?:of )?{MRE}\b"):
        if m[2] == "may" and not text[m.start(2):m.start(2) + 1].isupper():
            continue
        x = _infer_year_day(MON_ALL[m[2]], int(m[1]), anchor, fut(m))
        add(m, "day", x, x)
    for m in it(r"\b(?:on|the|since|from|until|by) (?:the )?(\d{1,2})/(\d{1,2})\b"):
        x = _infer_year_day(int(m[1]), int(m[2]), anchor) if 1 <= int(m[1]) <= 12 else None
        add(m, "day", x, x)
    for m in it(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b(?! (?:century|time|anniversary|birthday|grade|place|floor))"):
        d = int(m[1])
        if 1 <= d <= 31:
            x = _safe_date(anchor.year, anchor.month, d) if d <= calendar.monthrange(anchor.year, anchor.month)[1] else None
            if x and x > anchor + D(days=3):
                y, mo = add_months(anchor, -1)
                x = _safe_date(y, mo, d)
            add(m, "day", x, x)
    for m in it(rf"\b{MRE},? ((?:19|20)\d\d)\b"):
        if m[1] == "may" and not text[m.start():m.start() + 1].isupper():
            continue
        lo, hi = month_iv(int(m[2]), MON_ALL[m[1]])
        add(m, "month", lo, hi)
    for m in it(r"\b(last|this|next) (summer|winter|spring|fall|autumn)\b"):
        y = anchor.year + {"last": -1, "this": 0, "next": 1}[m[1]]
        if m[1] == "last" and m[2] in ("summer", "spring") and anchor.month >= 10:
            y = anchor.year
        lo, hi = season_iv(y, m[2])
        add(m, "season", lo, hi)
    for m in it(r"\b(summer|winter|spring|fall|autumn),? (?:of )?((?:19|20)\d\d)\b"):
        lo, hi = season_iv(int(m[2]), m[1])
        add(m, "season", lo, hi)
    for m in it(rf"\b(last|next|this|early|mid|late|in|during|since|by|until|around|of|end of|beginning of|start of) (?:the month of )?{MRE}\b(?! \d)"):
        word = m[2]
        if word == "may" and not text[m.start(2):m.start(2) + 1].isupper():
            continue
        if word in ("mar", "jun", "jan", "dec", "sep") and not text[m.start(2):m.start(2) + 1].isupper():
            continue  # short forms only when capitalised
        mo = MON_ALL[word]
        y = _infer_year_month(mo, anchor, m[1] if m[1] in ("last", "next") else "in")
        lo, hi = month_iv(y, mo)
        if m[1] == "early":
            hi = lo + D(days=10)
        elif m[1] == "mid":
            lo, hi = lo + D(days=9), lo + D(days=20)
        elif m[1] in ("late", "end of"):
            lo = hi - D(days=10)
        add(m, "month", lo, hi, group=0)
    for m in it(r"\b(?:in|since|from|by|around|until|of|back in|during) ((?:19|20)\d\d)\b"):
        y = int(m[1])
        add(m, "year", dt.date(y, 1, 1), dt.date(y, 12, 31))
    # ---- relative to the anchor
    for m in it(r"\b(?:the )?day before yesterday\b"):
        x = anchor - D(days=2)
        add(m, "day", x, x)
    for m in it(r"\b(?:the )?day after tomorrow\b"):
        x = anchor + D(days=2)
        add(m, "day", x, x)
    for m in it(r"\byesterday\b|\blast night\b"):
        x = anchor - D(days=1)
        add(m, "day", x, x)
    for m in it(r"\btoday\b|\btonight\b|\bthis (?:morning|afternoon|evening)\b"):
        add(m, "day", anchor, anchor)
    for m in it(r"\btomorrow\b"):
        x = anchor + D(days=1)
        add(m, "day", x, x)
    for m in it(rf"\b{NRE} (day|week|month|year)s? (?:ago|back|earlier)\b"):
        n = _n(m[1])
        if n is None:
            continue
        u = m[2]
        vague = m[1] in ("a few", "few", "several", "a couple of", "couple of", "a couple", "couple") and u == "day"
        if u == "day":
            x = anchor - D(days=n)
            if vague:
                add(m, "few_days", anchor - D(days=6), anchor - D(days=1), vague=True)
            else:
                add(m, "day", x, x)
        elif u == "week":
            x = anchor - D(days=7 * n)
            add(m, "weeks_ago", x - D(days=max(1, n)), x + D(days=max(1, n)))
        elif u == "month":
            y, mo = add_months(anchor, -n)
            lo, hi = month_iv(y, mo)
            add(m, "month", lo, hi)
        elif n == 1:  # "a year ago": month-level, centred on the anniversary
            x = _safe_date(anchor.year - 1, anchor.month, min(anchor.day, 28))
            add(m, "year_ago", x - D(days=15), x + D(days=15))
        else:
            add(m, "year", dt.date(anchor.year - n, 1, 1), dt.date(anchor.year - n, 12, 31))
    for m in it(r"\bthe other day\b|\bthe other night\b"):
        add(m, "few_days", anchor - D(days=6), anchor - D(days=1), vague=True)
    for m in it(r"\b(?:last|this past|past) weekend\b|\bover the weekend\b|\bon the weekend\b|\bthis weekend just gone\b"):
        sat = anchor - D(days=((anchor.weekday() - 5) % 7) or 7)
        add(m, "weekend", sat, sat + D(days=1))
    for m in it(r"\b(?:this|next|this coming) weekend\b"):
        sat = anchor + D(days=((5 - anchor.weekday()) % 7) or 7)
        add(m, "weekend", sat, sat + D(days=1))
    for m in it(r"\blast week\b|\bthe other week\b|\bthe past week\b|\ba week ago\b|\bthe previous week\b"):
        add(m, "week", anchor - D(days=7), anchor - D(days=1))
    for m in it(r"\bnext week\b"):
        add(m, "week", anchor + D(days=1), anchor + D(days=7))
    for m in it(r"\bthis week\b"):
        add(m, "week", anchor - D(days=anchor.weekday()), anchor)
    for m in it(r"\blast month\b|\bthe previous month\b"):
        lo, hi = month_iv(*add_months(anchor, -1))
        add(m, "month", lo, hi)
    for m in it(r"\bnext month\b"):
        lo, hi = month_iv(*add_months(anchor, 1))
        add(m, "month", lo, hi)
    for m in it(r"\bthis month\b"):
        add(m, "month", anchor.replace(day=1), anchor)
    for m in it(r"\blast year\b"):
        add(m, "year", dt.date(anchor.year - 1, 1, 1), dt.date(anchor.year - 1, 12, 31))
    for m in it(r"\bnext year\b"):
        add(m, "year", dt.date(anchor.year + 1, 1, 1), dt.date(anchor.year + 1, 12, 31))
    for m in it(r"\bthis year\b"):
        add(m, "year", dt.date(anchor.year, 1, 1), anchor)
    for m in it(r"\b(last|this past|past|on|this|next|this coming) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"):
        wd = WD.index(m[2])
        if m[1] in ("next", "this coming"):
            x = anchor + D(days=((wd - anchor.weekday()) % 7) or 7)
            add(m, "weekday", x, x)
        else:
            x = anchor - D(days=((anchor.weekday() - wd) % 7) or 7)
            add(m, "weekday", x, x)
    for m in it(r"(?<!\w)(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?!s)\b"):
        if weekday_in_name(text, m.start()) or explicit_date_follows(text, m.end(), out):
            claim(m)
            continue
        wd = WD.index(m[1])
        x = anchor - D(days=((anchor.weekday() - wd) % 7) or 7)
        add(m, "weekday_bare", x, x)
    for m in it(rf"\b(?:for|been|it's been|now|over|almost|nearly) (?:about |almost |nearly |over |the past |the last )?{NRE} years?\b|\b{NRE} years now\b"):
        n = _n(m[1] or m[2])
        if n:
            add(m, "duration_start_year", dt.date(anchor.year - n, 1, 1), dt.date(anchor.year - n, 12, 31))
    for m in it(rf"\b(?:for|been) (?:about |almost |nearly |over |the past |the last )?{NRE} (month|week)s?(?: now)?\b"):
        n = _n(m[1])
        if not n:
            continue
        if m[2] == "month":
            lo, hi = month_iv(*add_months(anchor, -n))
            add(m, "duration_start_month", lo, hi)
        else:
            x = anchor - D(days=7 * n)
            add(m, "duration_start_week", x - D(days=3), x + D(days=3))
    for m in it(r"\b(?:a|my|the|our|this|that|great|fun|nice|good|long|busy|super fun|relaxing) (?:\w+ )?weekend\b"):
        if re.search(r"(next|coming|upcoming|plan|will|going to|gonna)\W+(?:\w+\W+){0,3}$", t[max(0, m.start() - 40):m.start()]):
            continue
        sat = anchor - D(days=((anchor.weekday() - 5) % 7) or 7)
        add(m, "weekend_bare", sat, sat + D(days=1))
    out.sort(key=lambda e: e["start"])
    return out


# ------------------------------------------------------------------ rendered forms

RESOLVER_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
YEAR = re.compile(r"\b(?:19|20)\d\d\b")


def long_date(d: dt.date) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def iso_day(d: dt.date) -> str:
    return f"{d.isoformat()} ({WD[d.weekday()].capitalize()})"


def calendar_form(lo: dt.date, hi: dt.date) -> str:
    if lo == hi:
        return iso_day(lo)
    if lo.day == 1 and hi == month_iv(lo.year, lo.month)[1]:
        return f"{calendar.month_name[lo.month]} {lo.year}"
    if (lo.month, lo.day, hi.month, hi.day) == (1, 1, 12, 31) and lo.year == hi.year:
        return str(lo.year)
    return f"{lo.isoformat()} to {hi.isoformat()}"


def _count(n: int, unit: str) -> str:
    return f"{n} {unit}{'' if n == 1 else 's'}"


def template_form(e: dict, anchor: dt.date) -> str | None:
    """The answer-style form anchored on the session date, e.g. "the week before 8 May 2023"."""
    lo, hi, kind, a = e["lo"], e["hi"], e["kind"], long_date(anchor)
    side = "before" if hi < anchor else "after" if lo > anchor else None
    if kind == "day":
        delta = (lo - anchor).days
        if delta == 0:
            return None
        if abs(delta) == 1:
            return f"the day {side} {a}"
        return f"{_count(abs(delta), 'day')} {side} {a}"
    if kind in ("weekday", "weekday_bare") and side:
        return f"the {WD[lo.weekday()].capitalize()} {side} {a}"
    if kind in ("weekend", "weekend_bare") and side:
        return f"the weekend {side} {a}"
    if kind == "week":
        return f"the week {side} {a}" if side else f"the week of {a}"
    if kind == "weeks_ago":
        n = round((anchor - (lo + (hi - lo) / 2)).days / 7)
        return f"{_count(n, 'week')} before {a}"
    if kind == "few_days":
        return f"a few days before {a}"
    if kind == "year_ago":
        return f"a year before {a}"
    if kind == "month" and side:
        months = (anchor.year - lo.year) * 12 + anchor.month - lo.month
        if lo.day == 1 and abs(months) == 1:
            return f"the month {side} {a}"
        return None
    if kind == "year" and side and lo.year != anchor.year:
        return f"{_count(abs(anchor.year - lo.year), 'year')} {side} {a}" if abs(anchor.year - lo.year) > 1 else f"the year {side} {a}"
    if kind.startswith("duration_start"):
        return f"since about {calendar_form(lo, hi)} (counted back from {a})"
    return None


def is_relative(e: dict) -> bool:
    """True when the phrase needs the session date to be understood (no explicit year in it)."""
    return not YEAR.search(e["text"])


def annotate(e: dict, anchor: dt.date, template: bool = True) -> str | None:
    """The inline annotation for one expression: "[= 2023-05-07 (Sunday); the day before 8 May 2023]"."""
    if not is_relative(e):
        return None
    if e["kind"] == "day" and e["lo"] == anchor:
        return f"[= {iso_day(anchor)}]"
    parts = [calendar_form(e["lo"], e["hi"])]
    if template:
        t = template_form(e, anchor)
        if t:
            parts.append(t)
    return "[= " + "; ".join(parts) + "]"


def parse_iso_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s[:10])


def explicit_interval(when: str) -> tuple[dt.date, dt.date] | None:
    """A human-set date `@YYYY`, `@YYYY-MM` or `@YYYY-MM-DD` in a fact line (overrides the resolver)."""
    m = re.fullmatch(r"@(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", when.strip())
    if not m:
        return None
    y = int(m[1])
    if m[3]:
        d = _safe_date(y, int(m[2]), int(m[3]))
        return (d, d) if d else None
    if m[2]:
        return month_iv(y, int(m[2])) if 1 <= int(m[2]) <= 12 else None
    return dt.date(y, 1, 1), dt.date(y, 12, 31)
