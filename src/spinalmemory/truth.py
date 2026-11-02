"""The Markdown truth files. Everything Spine knows is in these files; the graph is derived from them.

One file per session, `sessions/<slug>.md`:

    ---
    session: session_3
    date: 2023-05-08T13:56
    raw-date: 1:56 pm on 8 May, 2023
    ---

    ### D3:1
    speaker: Priya
    role: speaker
    caption: a photo of a beagle on a beach
    > Hi Tom! I joined a pottery class yesterday.
    - m | Priya joined a pottery class yesterday. | Priya | hobby | pottery | "yesterday"
    - h | Priya lives in Leeds. | Priya | lives_in | Leeds | - | also: D1:3

- A turn block starts with `### <turn id>`. Ids are stable and never renumbered; an inserted turn gets a fresh id.
  The block order in the file is the turn order. `said-at` is the session date unless the block overrides it.
- Turn text is quoted line by line with `> `, so any text (including Markdown lists) round-trips.
- Fact lines: `- <tag> | sentence | subject | attribute | value | when [| also: <turn id> ...]`.
  tag `m` = machine-written (replaced on re-ingest), `h` = human-written (always kept).
  when = `"raw time phrase"` (resolved by code, never by the writer), `@YYYY[-MM[-DD]]` (a human-set date that
  overrides the resolver), or `-`. `also:` lists further source turns (facts are n:m with turns).
- Fact ids come from the file: `<turn id>#m01`, `<turn id>#h01` (position among that turn's lines of that tag).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TAGS = ("m", "h")


@dataclass(frozen=True)
class FactLine:
    tag: str
    sentence: str
    subject: str
    attribute: str
    value: str
    when: str = "-"
    also: tuple[str, ...] = ()

    def render(self) -> str:
        fields = [self.tag, self.sentence, self.subject, self.attribute, self.value, self.when]
        out = "- " + " | ".join(clean_field(f) for f in fields)
        if self.also:
            out += " | also: " + " ".join(self.also)
        return out


@dataclass
class TurnBlock:
    id: str
    speaker: str                 # the person's name, or "User" / "Assistant"
    role: str                    # "speaker" (named speakers), "user" or "assistant"
    text: str
    caption: str | None = None
    said_at: str | None = None   # None: the session date
    facts: list[FactLine] = field(default_factory=list)

    def fact_ids(self) -> list[str]:
        """Ids of this turn's facts, in file order."""
        seen = {t: 0 for t in TAGS}
        out = []
        for f in self.facts:
            seen[f.tag] += 1
            out.append(f"{self.id}#{f.tag}{seen[f.tag]:02d}")
        return out


@dataclass
class SessionFile:
    session_id: str
    date: str                    # ISO "YYYY-MM-DDTHH:MM"
    raw_date: str = ""
    turns: list[TurnBlock] = field(default_factory=list)

    def render(self) -> str:
        lines = ["---", f"session: {self.session_id}", f"date: {self.date}"]
        if self.raw_date:
            lines.append(f"raw-date: {one_line(self.raw_date)}")
        lines += ["---", ""]
        for t in self.turns:
            lines.append(f"### {t.id}")
            lines.append(f"speaker: {one_line(t.speaker)}")
            lines.append(f"role: {t.role}")
            if t.said_at:
                lines.append(f"said-at: {t.said_at}")
            if t.caption is not None:
                lines.append(f"caption: {one_line(t.caption)}")
            for part in t.text.split("\n"):
                lines.append("> " + part if part else ">")
            for f in t.facts:
                lines.append(f.render())
            lines.append("")
        return "\n".join(lines)


def one_line(s: str) -> str:
    return re.sub(r"\s*\n\s*", " ", s).strip()


def clean_field(s: str) -> str:
    """Fact fields cannot hold the separator or a newline."""
    return one_line(s).replace("|", "/")


def session_slug(session_id: str) -> str:
    """A file name for a session id (ids are kept verbatim inside the file)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", session_id) + ".md"


class TruthFormatError(ValueError):
    pass


FACT = re.compile(r"^- ([mh]) \| (.*)$")


def parse_fact(line: str) -> FactLine:
    m = FACT.match(line)
    if not m:
        raise TruthFormatError(f"not a fact line: {line!r}")
    parts = [p.strip() for p in m.group(2).split(" | ")]
    also: tuple[str, ...] = ()
    if len(parts) == 6 and parts[5].startswith("also:"):
        also = tuple(parts[5][5:].split())
        parts = parts[:5]
    if len(parts) != 5:
        raise TruthFormatError(f"fact line needs 5 fields after the tag: {line!r}")
    sentence, subject, attribute, value, when = parts
    return FactLine(m.group(1), sentence, subject, attribute, value, when or "-", also)


def parse_session(text: str) -> SessionFile:
    lines = text.split("\n")
    if not lines or lines[0] != "---":
        raise TruthFormatError("missing front matter")
    end = lines.index("---", 1)
    meta = dict(line.split(": ", 1) for line in lines[1:end] if ": " in line)
    sess = SessionFile(meta["session"], meta["date"], meta.get("raw-date", ""))
    cur: TurnBlock | None = None
    text_lines: list[str] = []

    def close() -> None:
        if cur is not None:
            cur.text = "\n".join(text_lines)
            sess.turns.append(cur)

    for line in lines[end + 1:]:
        if line.startswith("### "):
            close()
            cur = TurnBlock(id=line[4:].strip(), speaker="", role="", text="")
            text_lines = []
        elif cur is None or line == "":
            continue
        elif line.startswith(">"):
            text_lines.append(line[2:] if line.startswith("> ") else line[1:])
        elif line.startswith("- "):
            cur.facts.append(parse_fact(line))
        elif line.startswith("speaker: "):
            cur.speaker = line[9:]
        elif line.startswith("role: "):
            cur.role = line[6:]
        elif line.startswith("said-at: "):
            cur.said_at = line[9:]
        elif line.startswith("caption: "):
            cur.caption = line[9:]
        else:
            raise TruthFormatError(f"unexpected line in turn {cur.id}: {line!r}")
    close()
    return sess


def read_store(store: Path) -> dict[str, SessionFile]:
    """All session files of a store, keyed by session id."""
    out = {}
    for p in sorted((store / "sessions").glob("*.md")):
        s = parse_session(p.read_text(encoding="utf-8"))
        out[s.session_id] = s
    return out


def write_session(store: Path, sess: SessionFile) -> Path:
    path = store / "sessions" / session_slug(sess.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(sess.render(), encoding="utf-8")
    tmp.replace(path)
    return path
