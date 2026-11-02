"""The size policy: one memory-token budget rule for every question (`pack.policy`).

    fixed     B = pack.budget
    adaptive  B = pack.budget_small  if the top-1 cross-encoder score leads the k-th by at least margin_threshold
              B = pack.budget        otherwise

The scores are those of the one scorer over the one pool. A pool shorter than k counts as confident. Temporal
and advice questions use the same rule. The rule reads only sorted floats: no ids and no labels.
"""

from __future__ import annotations


def margin(scores: list[float], k: int) -> float | None:
    """score[0] - score[k-1] of scores sorted highest first; None when there are fewer than k."""
    return scores[0] - scores[k - 1] if len(scores) >= k else None


def choose(pack: dict, scores: dict[str, float]) -> tuple[int, float | None]:
    """(budget, margin) for one question. `pack` is the config's [pack] section."""
    m = margin(sorted(scores.values(), reverse=True), int(pack["margin_k"]))
    if pack["policy"] == "fixed":
        return int(pack["budget"]), m
    if pack["policy"] != "adaptive":
        raise ValueError(f"unknown pack.policy {pack['policy']!r}")
    small = m is None or m >= float(pack["margin_threshold"])
    return int(pack["budget_small"] if small else pack["budget"]), m
