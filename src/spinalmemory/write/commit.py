"""Commit validated lines to the truth files. Only machine-tagged lines are replaced; human lines are kept."""

from __future__ import annotations

import copy

from spinalmemory.truth import FactLine, SessionFile


def commit(sess: SessionFile, lines: dict[str, list[FactLine]], previous: SessionFile | None = None) -> SessionFile:
    """A copy of `sess` whose turns carry the new machine lines, plus the human lines `previous` had for them."""
    human = {}
    if previous is not None:
        for t in previous.turns:
            human[t.id] = [f for f in t.facts if f.tag == "h"]
    out = copy.deepcopy(sess)
    for t in out.turns:
        t.facts = list(lines.get(t.id, [])) + human.get(t.id, [])
    return out
