"""The edit router, for a person editing a store's Markdown.

- Fact-line edit or addition only -> incremental projection (0 LLM calls).
- Turn text, speaker or caption changed, or turns added/removed -> that session is re-ingested (1 writer call,
  cached by window), human-tagged lines are kept, then the incremental projection.
"""

from __future__ import annotations

import fcntl
from pathlib import Path

from spinalmemory.project.projection import Projection
from spinalmemory.project.vectors import Encoder
from spinalmemory.truth import SessionFile, read_store, write_session
from spinalmemory.write.build import codecs_for, materialise_vectors
from spinalmemory.write.commit import commit
from spinalmemory.write.windows import make_window
from spinalmemory.write.writer import write_window


def turns_changed(old: SessionFile | None, new: SessionFile) -> bool:
    """True if the writer's input changed (a session-date edit alone only re-projects)."""
    if old is None:
        return True
    a = [(t.id, t.speaker, t.role, t.text, t.caption) for t in old.turns]
    b = [(t.id, t.speaker, t.role, t.text, t.caption) for t in new.turns]
    return a != b


async def apply_edit(store: Path, new: SessionFile, cfg: dict) -> dict:
    """Route one edited session file; returns what was done."""
    proj = Projection.load(store / "index" / "projection.pkl", codecs_for(cfg))
    old = read_store(store).get(new.session_id)
    calls = 0
    if turns_changed(old, new):
        b = cfg["build"]
        from spinalmemory import llm

        tok = llm.tokenizer() if proj.benchmark == "longmemeval" else None
        window = make_window(new, proj.benchmark, b["lme_turn_max_tokens"], tok)
        val, rec, cached = await write_window(window, b["writer_prompt"], b["writer_max_tokens"], b["max_lines_per_turn"],
                                              llm.settings(cfg, "writer"))
        calls = 0 if cached else len(rec["attempts"])
        new = commit(new, val.lines, previous=new)
    with open(store / ".project.lock", "a") as lock_file:  # readers (OpenStore) wait while the index changes
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        write_session(store, new)
        counts = proj.update({new.session_id: new})
        materialise_vectors(proj, store / "index", Encoder(cfg["build"]["encoder"], cfg["device"]["encoder"]))
        proj.save(store / "index" / "projection.pkl")
        (store / "index" / "dump.sha256").write_text(proj.dump_hash() + "\n")
    return {"writer_calls": calls, **counts}
