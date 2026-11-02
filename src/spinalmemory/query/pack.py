"""Packing: whole turns into a fixed budget B of rendered memory tokens.

- Episode completion (off by default): after each of the top `seeds` turns, its previous 1 and next 2
  same-session turns of at most `cap` tokens (inside the partition) are placed right behind it; they are drawn
  from inside B.
- A turn's planned cost is its turn line plus, when fact lines are rendered, its fact-line cost.
- Greedy fill: turns in order; a turn that does not fit is skipped and smaller later turns may still fit.
  A session's header is charged when its first turn enters.
- Exact fill (`fill`): the plan does not see the fact-history section (its lines depend on which turns are packed)
  or the folding of confirmed and duplicate fact lines, so the exact count of the rendered block differs from the
  plan. `fill` searches the plan budget: it plans, renders and counts exactly, moves the plan budget by the
  measured difference (bracketed between the largest plan budget known to fit and the smallest known to overshoot),
  and keeps the packing with the largest exact count that fits. It stops once the room left is at most
  FILL_TOLERANCE of B (or no unpacked turn's plan cost fits in it), after at most FILL_ROUNDS exact renders.
  The result never exceeds B and lands close to it.
"""

from __future__ import annotations

from typing import Callable, TypeVar

R = TypeVar("R")

FILL_TOLERANCE = 0.01  # stop when at most 1% of the budget is left (each extra render costs retrieval time)
FILL_ROUNDS = 6        # exact renders before settling for the fullest packing that fits


def episode_complete(order: list[str], session_of: Callable[[str], list[str]], cost: dict[str, int],
                     seeds: int = 10, prev: int = 1, nxt: int = 2, cap: int = 80) -> list[str]:
    seed_set = set(order[:seeds])
    out, seen = [], set()
    for t in order:
        if t in seen:
            continue
        out.append(t)
        seen.add(t)
        if t in seed_set:
            ss = session_of(t)
            i = ss.index(t)
            for u in ss[max(0, i - prev):i] + ss[i + 1:i + 1 + nxt]:
                if u not in seen and cost[u] <= cap:
                    out.append(u)
                    seen.add(u)
    return out


def greedy(order: list[str], cost: dict[str, int], budget: float, session: dict[str, str],
           header_cost: dict[str, int], start: list[str] | None = None) -> list[str]:
    out = list(start or [])
    open_sessions = {session[t] for t in out}
    used = sum(cost[t] for t in out) + sum(header_cost[s] for s in open_sessions)
    have = set(out)
    for t in order:
        if t in have:
            continue
        c = cost[t] + (0 if session[t] in open_sessions else header_cost[session[t]])
        if used + c <= budget:
            out.append(t)
            have.add(t)
            used += c
            open_sessions.add(session[t])
    return out


def planned(packed: list[str], cost: dict[str, int], session: dict[str, str], header_cost: dict[str, int]) -> int:
    """The plan's cost of a packing: its turns plus one header per session."""
    return sum(cost[t] for t in packed) + sum(header_cost[s] for s in {session[t] for t in packed})


def fill(order: list[str], cost: dict[str, int], budget: int, session: dict[str, str], header_cost: dict[str, int],
         measure: Callable[[list[str]], tuple[R, int]], max_rounds: int = FILL_ROUNDS,
         tolerance: float = FILL_TOLERANCE) -> tuple[list[str], R, int, int]:
    """Pack to the exact budget (module docstring). `measure(packed)` renders and returns (result, exact tokens).
    Returns (packed, result, exact tokens, rounds). If no planned packing fits within `max_rounds` (a fact-history
    section alone near the budget), the smallest one tried is trimmed from its lowest-priority end until it fits
    (an empty packing renders an empty block, so this ends)."""
    lo, hi = None, None              # largest plan budget known to fit, smallest known to overshoot
    best = None                      # (tokens, packed, result) with the largest exact count <= budget
    smallest = None                  # (plan budget, packed, result, tokens) with the smallest plan budget tried
    tried: dict[tuple, int] = {}
    plan_b, rounds, steps = budget, 0, 0
    while rounds < max_rounds and steps < 4 * max_rounds:
        steps += 1
        packed = greedy(order, cost, plan_b, session, header_cost)
        key = tuple(packed)
        if key in tried:
            tok = tried[key]
        else:
            res, tok = measure(packed)
            rounds += 1
            tried[key] = tok
            if smallest is None or plan_b < smallest[0]:
                smallest = (plan_b, packed, res, tok)
            if tok <= budget and (best is None or tok > best[0]):
                best = (tok, packed, res)
        if tok <= budget:
            lo = plan_b if lo is None else max(lo, plan_b)
        else:
            hi = plan_b if hi is None else min(hi, plan_b)
        if best is not None:
            gap = budget - best[0]
            have = set(best[1])
            if gap <= tolerance * budget or all(cost[t] > gap for t in order if t not in have):
                break                    # nothing left could fit in the room that is left
        # the plan budget that would make the exact count land on the budget, kept inside the bracket
        nxt = plan_b + (budget - tok)
        if lo is not None and hi is not None:
            if hi - lo <= 1:
                break
            if not lo < nxt < hi:
                nxt = (lo + hi) // 2
        elif lo is not None and nxt <= lo:
            nxt = lo + 1
        elif hi is not None and nxt >= hi:
            nxt = hi - 1
        if nxt == plan_b or nxt < 0:
            break
        plan_b = nxt
    if best is not None:
        return best[1], best[2], best[0], rounds
    _, packed, res, tok = smallest
    while tok > budget and packed:   # trim lowest-priority turns: at least the overshoot in planned tokens
        over, k = tok - budget, 0
        while k < len(packed) - 1 and over > 0:
            k += 1
            over -= cost[packed[-k]]
        packed = packed[:-max(k, 1)]
        res, tok = measure(packed)
        rounds += 1
    return packed, res, tok, rounds
