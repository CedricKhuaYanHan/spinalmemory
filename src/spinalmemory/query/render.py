"""Render the one prompt (templated formatting by code; nothing is generated), and parse the one output.

The memory block is
  1. a fact-history section (when a matched spine has a packed member): one line per matched attribute spine,
     its values newest first, each with the handle of its source turn, its valid date when known and the date it
     was said; single-valued attributes mark the newest packed value current (if it is the spine head) and the
     others earlier; a human correction shows the value it replaced (the corrected machine value is not listed);
     a machine value that a human line confirms (same spine, cites its turn, same normalised value) is listed once,
     as the human value;
  2. sessions oldest first under ISO date headers (headers also say how many days before the question date when
     the question carries its own date; sessions with the same date and time follow a hash of their id); every
     relative time phrase is followed by its code-resolved date, or, on the temporal path, by its date and
     offset from the question date (query/temporal.py); short handles [1], [2], ... in reading order;
  3. under each packed turn, its fact lines with their valid dates (a corrected machine line is replaced by the
     human line, marked corrected; a confirmed machine line is shown once, as the human line; an identical
     rendered line is shown once). With render.facts = "added" only the human lines are shown there: the
     machine lines restate their passage, and the fact-history section is unchanged.
Enumerate-then-answer only for questions flagged as counting distinct items; the question at both ends; line
output "Answer:" / "Evidence:". The temporal and advice paths swap the instruction block (query/temporal.py,
query/advice.py).

The budget B counts the whole memory block (history section, headers, turn lines and fact lines); the
instruction block around it explains only the markers the block actually shows (`intro_notes`). Folding confirmed
and duplicate fact lines happens here, at render time: the projection's `fact_costs` (the plan cost) still counts
every line, so the plan errs high there, and the exact count in answer.py decides.

Output parsing (code over the one output): the Answer line is the hypothesis. Handles are cut from it (they
belong on the Evidence line; an answer of only handles becomes empty), and an output without an Answer line is
an empty answer when it was cut off at max_tokens or holds a Notes scratchpad; otherwise the raw output is the
answer.
"""

from __future__ import annotations

import datetime as dt
import re

from spinalmemory.project import entities as E
from spinalmemory.project.projection import norm_value
from spinalmemory.project.rendering import fact_line, turn_line
from spinalmemory.project.vocab import SINGLE_VALUED
from spinalmemory.project.timeexpr import iso_day
from spinalmemory.query import temporal as TQ
from spinalmemory.query.retrieve import tie_key

SYSTEM = "You answer questions about past conversations, using only the memory you are given. You answer briefly."

INTRO = ("Memory: turns from past conversations, sessions oldest first. [n] is a turn's handle. A bracket after a "
         "time phrase gives the date computed from that session's date.")
# each sentence is said only when the memory block holds the thing it explains
FACTS_NOTE = " The lines under a turn are facts noted from it, with the date each applies to (valid) when known."
HISTORY_NOTE = (" The fact history lists, for the people and attributes this question concerns, each value newest "
                "first with the turn it comes from, the date it applies to (valid) when known and the date it was "
                "said.")
CURRENT_NOTE = " A value marked current is the latest of something that holds one value at a time."
CORRECTED_NOTE = " A fact marked corrected was fixed by a person and replaces the value shown after \"replaced\"."
HISTORY_HEAD = "Fact history (newest first):"

OUTRO = ("Answer the question from the memory. Reply in exactly this format:\n"
         "Answer: <the answer, as short as possible>\n"
         "Evidence: <the handles of the turns you used, e.g. [3] [7]>")
# Enumerate-then-answer, kept on the Answer line so the answer stays first.
ENUMERATE = ("This question asks how many: on the Answer line, first list the items you counted, then the number, "
             "e.g. Answer: hiking, chess, piano = 3")


def session_header(date_iso: str, relative: bool, question_date: dt.date) -> str:
    """`relative` (input format: the question carries its own date) adds the day count before the question date."""
    d = dt.date.fromisoformat(date_iso[:10])
    head = f"## {iso_day(d)} {date_iso[11:16]}"
    if relative:
        n = (question_date - d).days
        rel = "on the question date" if n == 0 else f"{n} day{'s' if abs(n) != 1 else ''} before the question date" if n > 0 \
            else f"{-n} day{'s' if n != -1 else ''} after the question date"
        head += f", {rel}"
    return head


def _attr_label(attribute: str) -> str:
    return attribute.removeprefix("other:").replace("_", " ")


def _entity_label(proj, entity: str, members: list[str]) -> str:
    if entity == "user":
        return "User"
    for f in members:
        if E.norm_name(proj.facts[f]["subject"]) == entity:
            return proj.facts[f]["subject"]
    return entity


def confirmations(proj, f: str, humans: dict | None = None) -> list[str]:
    """The human lines that confirm machine fact `f`: same spine, citing f's turn, the same normalised value (the
    correction rule's condition with equal values, so they make no correction link). `humans` memoises each
    spine's human lines for one render."""
    fm = proj.facts[f]
    lane = proj.fact_lane.get(f)
    if fm["tag"] != "m" or lane is None:
        return []
    if humans is None:
        humans = {}
    if lane not in humans:
        humans[lane] = [g for g in proj.lane_members.get(lane, ()) if proj.facts[g]["tag"] == "h"]
    if not humans[lane]:
        return []
    val = norm_value(fm["value"])
    return [g for g in humans[lane] if fm["home"] in proj.facts[g]["sources"] and norm_value(proj.facts[g]["value"]) == val]


def facts_mode(value) -> bool | str:
    """The render.facts setting, checked: True (every fact line), False (none) or "added" (only the lines that add
    to their passage: human lines, corrections included)."""
    if value is True or value is False or value == "added":
        return value
    raise ValueError(f"render.facts must be true, false or \"added\", not {value!r}")


def turn_fact_lines(proj, tid: str, humans: dict | None = None, added_only: bool = False) -> tuple[list[str], int]:
    """The fact lines shown under a turn and how many of them show a correction. As `proj.fact_lines` (the plan cost),
    except that a machine line confirmed by a human line also shown here is folded into it, and an identical
    rendered line is shown once.
    `added_only` (render.facts = "added") keeps only the human lines: a machine line is the writer's restatement
    of the passage it sits under, while a human line, a correction among them (shown with the value it replaced),
    adds what the passage does not say. A corrected machine line is never shown either way (`turn_facts`)."""
    facts = proj.turn_facts(tid)
    here = set(facts)
    out, seen, corr = [], set(), 0
    for f in facts:
        if added_only and proj.facts[f]["tag"] != "h":
            continue
        if any(g in here for g in confirmations(proj, f, humans)):
            continue
        replaced = [proj.facts[m]["value"] for m in proj.corrects.get(f, ())]
        line = fact_line(proj.facts[f]["sentence"], proj.valid_text(f), replaced)
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
        corr += bool(replaced)
    return out, corr


def added_fact_cost(proj, tid: str, count_tokens) -> int:
    """The plan cost of a turn's fact lines under render.facts = "added" (as `fact_costs`, over the human lines
    only; a turn without a human line costs 0 and needs no count)."""
    if not any(proj.facts[f]["tag"] == "h" for f in proj.turn_facts(tid)):
        return 0
    lines, _ = turn_fact_lines(proj, tid, added_only=True)
    return count_tokens("\n".join(lines)) + 1 if lines else 0


def history_lines(proj, spines: list[tuple[str, str]], handle_of: dict[str, int], max_values: int,
                  humans: dict | None = None) -> tuple[list[str], dict]:
    """The fact-history section: one line per matched spine with a packed member (module docstring). Returns the
    lines and the markers they show ({"current", "corrected"} counts), so the intro explains only those."""
    out = []
    humans = {} if humans is None else humans
    shown = {"current": 0, "corrected": 0}
    for key in spines:
        sp = proj.spines.get(key)
        if not sp:
            continue
        single = key[1] in SINGLE_VALUED
        # the head, and the human lines that confirm it (shown in its place when it is folded into them)
        current = {sp["head"], *confirmations(proj, sp["head"], humans)} if single and sp["head"] else set()
        vals = []
        for f in sp["members"]:
            if f in proj.corrected:  # a corrected machine value is shown only as "replaced"
                continue
            if any(t in handle_of for g in confirmations(proj, f, humans) for t in proj.fact_src.get(g, ())):
                continue  # confirmed by a human line that is shown: listed once, as the human value
            hs = sorted(handle_of[t] for t in proj.fact_src.get(f, ()) if t in handle_of)
            if not hs:
                continue
            fr = proj.facts[f]
            notes = []
            if single:
                notes.append("current" if f in current else "earlier")
            if proj.corrects.get(f):
                notes.append("corrected; replaced: " + "; ".join(proj.facts[m]["value"] for m in proj.corrects[f]))
            v = proj.valid_text(f)
            if v:
                notes.append(f"valid {v}")
            notes.append(f"said {fr['said_at'][:10]}")
            val = f"{fr['value']} [{hs[0]}] ({', '.join(notes)})"
            if val in vals:  # two members that render identically (the same value, turn and dates): shown once
                continue
            vals.append(val)
            shown["current"] += f in current
            shown["corrected"] += bool(proj.corrects.get(f))
            if len(vals) >= max_values:
                break
        if vals:
            out.append(f"- {_entity_label(proj, key[0], sp['members'])}, {_attr_label(key[1])}: " + "; ".join(vals))
    return out, shown


def render_memory(store, packed: list[str], question_date: dt.date, template: bool, relative: bool = False,
                  facts: bool | str = False, spines: list | None = None, history_values: int = 8,
                  question_date_brackets: bool = False) -> tuple[str, dict[int, str], dict]:
    """The memory block, its handles and what it shows ({"history_lines", "fact_lines", "corrected_shown",
    "history_current", "history_corrected"}).
    `question_date_brackets` (temporal path) dates each phrase against the question date; `relative` adds day
    counts to the session headers; `facts` adds fact lines under turns (True: all of them; "added": only the
    human lines, corrections included; see `turn_fact_lines`); `spines` (matched attribute spines) adds the
    fact-history section, which does not depend on `facts`."""
    proj = store.proj
    added_only = facts == "added"
    by_session: dict[str, list[str]] = {}
    for t in packed:
        by_session.setdefault(proj.turns[t]["session"], []).append(t)
    order = sorted(by_session, key=lambda s: (proj.sessions[s]["date"], tie_key(s)))
    lines, handles, n = [], {}, 0
    n_facts = n_corr = 0
    humans: dict = {}  # spine -> its human lines, for this render
    for sid in order:
        if lines:
            lines.append("")
        lines.append(session_header(proj.sessions[sid]["date"], relative, question_date))
        for t in sorted(by_session[sid], key=lambda x: proj.turns[x]["pos"]):
            n += 1
            handles[n] = t
            r = proj.turns[t]
            if question_date_brackets:
                text = TQ.annotated_text(r["text"], proj.timex.get(t, []), question_date)
            else:
                text = proj.rendered_text(t, template)
            lines.append(turn_line(str(n), r["speaker"], text, r["caption"]))
            if facts:
                fl, corr = turn_fact_lines(proj, t, humans, added_only)
                n_facts += len(fl)
                n_corr += corr
                lines += fl
    hist, marks = (history_lines(proj, spines, {t: h for h, t in handles.items()}, history_values, humans) if spines
                   else ([], {"current": 0, "corrected": 0}))
    if hist:
        lines = [HISTORY_HEAD, *hist, ""] + lines
    return "\n".join(lines), handles, {"history_lines": len(hist), "fact_lines": n_facts, "corrected_shown": n_corr,
                                       "history_current": marks["current"], "history_corrected": marks["corrected"]}


def intro_notes(info: dict) -> str:
    """The intro sentences for what the memory block holds (nothing is explained that is not shown): fact lines,
    the fact history, a `current` marker, a `corrected` marker (in a fact line or the fact history)."""
    return ((FACTS_NOTE if info.get("fact_lines") else "") + (HISTORY_NOTE if info.get("history_lines") else "")
            + (CURRENT_NOTE if info.get("history_current") else "")
            + (CORRECTED_NOTE if info.get("corrected_shown") or info.get("history_corrected") else ""))


def question_date_line(own_date: bool, question_date_iso: str) -> str:
    """`own_date` (input format): the question carries its own asking date."""
    d = dt.date.fromisoformat(question_date_iso[:10])
    if own_date:
        return f"Question date: {iso_day(d)}"
    return f"Date of the last conversation: {iso_day(d)}"


def render_prompt(question: str, memory: str, date_line: str, notes: str = "", counting: bool = False,
                  advice: bool = False) -> str:
    """The standard prompt; `advice` swaps the instruction block for the advice block; `notes` = intro_notes()."""
    from spinalmemory.query.advice import OUTRO as ADVICE_OUTRO

    intro = INTRO + notes
    outro = (ADVICE_OUTRO if advice else OUTRO) + ("\n" + ENUMERATE if counting else "")
    return f"Question: {question}\n\n{intro}\n\n{memory}\n\n{date_line}\nQuestion: {question}\n\n{outro}"


ANSWER = re.compile(r"^\W*answer\W*:\s*(.*)$", re.I)
EVIDENCE = re.compile(r"^\W*evidence\W*:\s*(.*)$", re.I)
NOTES = re.compile(r"^\W*notes\W*:", re.I)


def parse_output(text: str) -> tuple[str, list[int], bool]:
    """(hypothesis, evidence handles, had an Answer line). Without an Answer line the raw output is the answer.
    Lines after the Answer line join the answer only when the Answer line itself is empty (an answer written
    below its label); otherwise they are commentary and are dropped."""
    lines = text.strip().splitlines()
    ans, ev, found, in_answer = [], [], False, False
    for line in lines:
        m, e = ANSWER.match(line), EVIDENCE.match(line)
        if m and not found:
            found = True
            ans.append(m.group(1).strip(" *"))
            in_answer = not ans[-1]
        elif e or re.match(r"^\W*op\W*:", line, re.I):
            in_answer = False
            if not e:
                continue
            ev += [int(x) for x in re.findall(r"\[(\d+)\]", e.group(1))]
        elif in_answer and line.strip():
            ans.append(line.strip(" *"))
    if not found:
        return text.strip(), ev, False
    return " ".join(a for a in ans if a).strip(), ev, True


INLINE_EVIDENCE = re.compile(r"\s*\**\s*evidence\s*\**\s*:\s*(.*)$", re.I)
HANDLE = re.compile(r"\s*\[(\d+)\](?:\s*-\s*\[(\d+)\])?")


def strip_handles(answer: str) -> tuple[str, list[int]]:
    """Handles cut from an Answer line: an "Evidence: [..]" written on the
    line, and bare handles ("[34]", "[29]-[34]"). Returns the answer and the cut handles, which join the evidence.
    An answer with nothing to cut is returned unchanged."""
    m = INLINE_EVIDENCE.search(answer)
    if not m and not HANDLE.search(answer):
        return answer, []
    handles: list[int] = []
    if m:
        handles += [int(x) for x in re.findall(r"\[(\d+)\]", m.group(1))]
        answer = answer[: m.start()]
    for a, b in HANDLE.findall(answer):
        handles += [int(a)] + ([int(b)] if b else [])
    answer = HANDLE.sub("", answer)
    answer = re.sub(r"\s+([.,;:!?)])", r"\1", answer)
    return re.sub(r"\s{2,}", " ", answer).strip(), list(dict.fromkeys(handles))


def notes_of(text: str) -> str:
    """The lines before the Answer line (the temporal path's scratchpad), or the whole output when there is no Answer line."""
    lines = text.strip().splitlines()
    cut = next((i for i, ln in enumerate(lines) if ANSWER.match(ln)), len(lines))
    return "\n".join(lines[:cut]).strip()


def read_answer(text: str, truncated: bool, cut_handles: bool = True, empty_when_cut: bool = True) -> dict:
    """The parse of the one output: {"hypothesis", "evidence" (handles), "answer_line", "handles_cut", "emptied"}.

    - Answer line present: its text, with handles cut when `cut_handles` (they join the evidence handles).
    - No Answer line: an empty answer when the output was cut off at max_tokens or holds a Notes line (a scratchpad
      is never the answer), when `empty_when_cut`; otherwise the raw output."""
    hyp, ev, had = parse_output(text)
    cut: list[int] = []
    emptied = None
    if had and cut_handles:
        hyp, cut = strip_handles(hyp)
        ev = ev + [h for h in cut if h not in ev]
    elif not had and empty_when_cut and (truncated or any(NOTES.match(ln) for ln in text.splitlines())):
        hyp, emptied = "", "truncated" if truncated else "notes"
    return {"hypothesis": hyp, "evidence": ev, "answer_line": had, "handles_cut": cut, "emptied": emptied}
