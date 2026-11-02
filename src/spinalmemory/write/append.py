"""Append: turn a history into session files (turn blocks with stable ids, no fact lines yet).

Turn ids come from the input: a speaker-format turn keeps its `dia_id` ("D3:7"); a user/assistant-format turn
gets "<session_id>_<n>", with n the 1-based position of the turn in its session, counting all turns.
"""

from __future__ import annotations

import datetime as dt
import re

from spinalmemory.truth import SessionFile, TurnBlock


def iso_session_date(raw: str) -> str:
    """"1:56 pm on 8 May, 2023" or "2023/05/20 (Sat) 02:21" -> "2023-05-08T13:56"."""
    s = (raw or "").strip()
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y", "%I:%M %p on %d %B %Y"):
        try:
            return dt.datetime.strptime(s, fmt).strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            pass
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})(?: \(\w+\))?(?: (\d{2}):(\d{2}))?", s)
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}T{m[4] or '00'}:{m[5] or '00'}"
    raise ValueError(f"unparseable session date {raw!r}")


def history_sessions(history: dict) -> list[SessionFile]:
    out = []
    if history["benchmark"] == "locomo":
        for s in history["sessions"]:
            turns = [TurnBlock(id=t["dia_id"], speaker=t["speaker"], role="speaker", text=t["text"],
                               caption=t.get("image_caption")) for t in s["turns"]]
            out.append(SessionFile(s["session_id"], iso_session_date(s["date"]), s["date"], turns))
    else:
        seen = set()
        for s in history["sessions"]:
            if s["session_id"] in seen:  # a history should never repeat a session; guard anyway
                continue
            seen.add(s["session_id"])
            turns = [TurnBlock(id=f"{s['session_id']}_{i + 1}", speaker=t["role"].capitalize(), role=t["role"],
                               text=t["content"]) for i, t in enumerate(s["turns"])]
            out.append(SessionFile(s["session_id"], iso_session_date(s["date"]), s["date"], turns))
    return out
