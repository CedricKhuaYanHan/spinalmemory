"""Windows: one session per writer call.

A window is what the writer sees: short per-call ids (t1..tn) mapped to the session's turns. Speaker-format
windows hold every turn (with its photo caption); user/assistant-format windows hold the user turns only, each
truncated at 512 budget tokens. The window's text depends only on its own session, so a session shared by several
histories is one window, written once.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from spinalmemory.truth import SessionFile


@dataclass(frozen=True)
class Window:
    session_id: str
    ids: tuple[tuple[str, str], ...]   # (short id, turn id)
    lines: tuple[str, ...]             # "^t1 Priya: text [shares a photo: caption]"
    texts: dict                        # short id -> the text the writer saw (for literal-span checks)
    speakers: dict                     # short id -> speaker

    @property
    def body(self) -> str:
        return "\n".join(self.lines)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.body.encode()).hexdigest()


def _truncate(text: str, max_tokens: int, tok) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens]).rstrip() + " ..."


def make_window(sess: SessionFile, benchmark: str, lme_turn_max_tokens: int = 512, tok=None) -> Window:
    ids, lines, texts, speakers = [], [], {}, {}
    k = 0
    for t in sess.turns:
        if benchmark == "longmemeval" and t.role != "user":
            continue
        k += 1
        short = f"t{k}"
        text = " ".join(t.text.split()) if benchmark == "locomo" else t.text.strip()
        if benchmark == "longmemeval" and tok is not None:
            text = _truncate(text, lme_turn_max_tokens, tok)
        if t.caption:
            text += f" [shares a photo: {t.caption}]"
        ids.append((short, t.id))
        lines.append(f"^{short} {t.speaker}: {text}")
        texts[short] = text
        speakers[short] = t.speaker
    return Window(sess.session_id, tuple(ids), tuple(lines), texts, speakers)
