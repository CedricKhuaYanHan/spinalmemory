"""Entities and aliases, by code alone.

- A name is normalised by case-folding, collapsing whitespace and trimming punctuation.
- First-person subjects ("I", "me", "user") become the turn's speaker, or "user" in user/assistant histories.
- Fact lines with attribute `alias` ("Thomas | alias | Tom") merge two names. Merges are a union-find over the
  sorted pairs; each component's representative is its speaker name if it has one, else its smallest name.
  So the result depends only on the set of alias lines, never on the order they were read in.
"""

from __future__ import annotations

import re

ALIAS_RULES_VERSION = "1"
FIRST_PERSON = {"i", "me", "my", "myself", "user", "the user", "speaker"}


def norm_name(name: str) -> str:
    s = re.sub(r"\s+", " ", name.casefold()).strip()
    return s.strip(" .,:;!?\"'()[]")


def subject_name(subject: str, speaker: str, role: str) -> str:
    """The normalised subject of a fact line, with first person mapped to the turn's speaker."""
    s = norm_name(subject)
    if s in FIRST_PERSON or not s:
        return "user" if role in ("user", "assistant") else norm_name(speaker)
    return s


class UnionFind:
    def __init__(self, pairs: list[tuple[str, str]], preferred: set[str]):
        self.parent: dict[str, str] = {}
        self.preferred = preferred
        for a, b in sorted(pairs):
            self.union(a, b)

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def _rank(self, x: str) -> tuple[int, str]:
        return (0 if x in self.preferred else 1, x)

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        keep, drop = sorted((ra, rb), key=self._rank)
        self.parent[drop] = keep


def canonical_map(alias_pairs: list[tuple[str, str]], speakers: set[str]) -> dict[str, str]:
    """name -> representative, for every name that occurs in an alias pair."""
    uf = UnionFind(alias_pairs, speakers)
    return {name: uf.find(name) for name in sorted(uf.parent)}
