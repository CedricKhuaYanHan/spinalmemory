"""How one turn is rendered in the answer prompt. Shared by the projection (rendered token cost per turn) and
the renderer, so the cost the packer plans with is the cost of the line the model sees. The fact lines rendered
under a turn are costed apart (`fact_costs`)."""

from __future__ import annotations

import datetime as dt

from spinalmemory.project.timeexpr import annotate

COST_HANDLE = "00"  # handles are short ("[7]", "[23]"); costs are computed with a two-digit handle


def annotated_text(text: str, exprs: list[dict], anchor: dt.date, template: bool = True) -> str:
    """The turn text with each relative time phrase followed by its code-resolved date in brackets."""
    out, last = [], 0
    for e in sorted(exprs, key=lambda e: e["start"]):
        note = annotate(e, anchor, template)
        if note is None:
            continue
        out.append(text[last:e["end"]])
        out.append(" " + note)
        last = e["end"]
    out.append(text[last:])
    return "".join(out)


def turn_line(handle: str, speaker: str, text: str, caption: str | None, tag: str | None = None) -> str:
    head = f"[{handle}]" + (f" ({tag})" if tag else "")
    photo = f" [photo: {caption}]" if caption else ""
    return f"{head} {speaker}: {text}{photo}"


def fact_line(sentence: str, valid: str | None, replaced: list[str] | None = None) -> str:
    """A fact line under its turn: the sentence, its valid date when known, and for a human correction
    the machine value it replaced."""
    out = f"  - {sentence}"
    if valid:
        out += f" (valid {valid})"
    if replaced:
        out += " (corrected; replaced: " + "; ".join(replaced) + ")"
    return out
