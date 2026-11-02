"""Check fact history, by code (no LLM). Attribute spines are built by the projection (project/projection.py).

    seeds      the facts cited by the top `history.seed_k` first-stage turns (src_index: n:m fact <-> turn)
    matching   persons named in the question (canonical entities); in histories with user/assistant roles,
               also the user; if that leaves nobody, the subjects of the seed facts
    match      a spine holding a seed fact whose entity is a matching person and which has at least
               `history.min_members` facts
    additions  the source turns of a matching spine's members (inside the question's partition), in spine order
               (first seed rank), members newest first, a member's turns in first-stage order; they join the one
               rerank pool in answer.py

No order follows the spelling of a turn or session id. A seed turn's own fact lines come first, by suffix (#h01,
#m01, ...; they share the turn's id prefix), then the facts that cite it from other turns (`also:`), by the
label-free hash (`tie_key`). A member's source turns follow their first-stage rank, then the hash.

Corrected machine facts are skipped as seeds and as additions (their turn is still reached through the human
line that corrects them, which cites the same turn). The matched spines are recorded for the fact-history section.
"""

from __future__ import annotations

from spinalmemory.project.entities import norm_name
from spinalmemory.project.projection import rowid_of as tie_key  # = retrieve.tie_key (sha256 of the id)


def matching_persons(store, named: list[str], seed_facts: list[str]) -> set[str]:
    proj = store.proj
    persons = {proj._ent(norm_name(n)) for n in named}
    if store.has_roles:
        persons.add(proj._ent("user"))
    if not persons:
        persons = {proj.fact_ent[f][0] for f in seed_facts if f in proj.fact_ent}
    return persons


def check(store, first: list[str], named: list[str], part_set: set[str], seed_k: int, min_members: int
          ) -> tuple[list[tuple[str, str]], list[str], dict]:
    """(matched spine keys in seed order, spine-added turns in order, stats)."""
    proj = store.proj
    seeds: list[str] = []
    seen = set()
    for t in first[:seed_k]:
        cited = proj.src_index.get(t, ())
        own = sorted(f for f in cited if proj.facts[f]["home"] == t)        # same prefix: by suffix
        other = sorted((f for f in cited if proj.facts[f]["home"] != t), key=tie_key)
        for f in own + other:
            if f not in seen and f not in proj.corrected:
                seen.add(f)
                seeds.append(f)
    persons = matching_persons(store, named, seeds)
    spines: list[tuple[str, str]] = []
    for f in seeds:
        key = proj.fact_lane.get(f)
        if key is None or key in spines or key[0] not in persons:
            continue
        sp = proj.spines.get(key)
        if sp and len(sp["members"]) >= min_members:
            spines.append(key)
    rank = {t: i for i, t in enumerate(first)}
    unranked = len(rank)
    adds: list[str] = []
    have = set()
    for key in spines:
        for f in proj.spines[key]["members"]:
            if f in proj.corrected:
                continue
            for t in sorted(proj.fact_src.get(f, ()), key=lambda x: (rank.get(x, unranked), tie_key(x))):
                if t in part_set and t not in have:
                    have.add(t)
                    adds.append(t)
    return spines, adds, {"seed_facts": len(seeds), "matching_persons": sorted(persons)}


def one_pool(first: list[str], additions: list[str], limit: int, max_additions: int) -> tuple[list[str], list[str]]:
    """The one rerank pool with one shared limit that includes the spine additions.

    adds  = spine-added turns not already in first[:limit], capped at max_additions (and at the room left by the
            spine turns already inside first[:limit]);
    pool  = first[:limit] with every spine turn kept and only non-spine turns cut from its tail to make room,
            in first-stage order, then adds.
    A spine turn the first stage already ranked inside the limit is never cut to make room for another spine turn
    (a plain first[:limit - len(adds)] + adds would cut those ranked in the last len(adds) places).
    Returns (pool, adds); len(pool) <= limit always."""
    head = first[:limit]
    spine = set(additions)
    in_head = set(head)
    kept = sum(1 for t in head if t in spine)
    adds = [t for t in additions if t not in in_head][:max(0, min(max_additions, limit - kept))]
    room = limit - kept - len(adds)  # non-spine turns that stay
    pool, others = [], 0
    for t in head:
        if t in spine:
            pool.append(t)
        elif others < room:
            pool.append(t)
            others += 1
    return pool + adds, adds
