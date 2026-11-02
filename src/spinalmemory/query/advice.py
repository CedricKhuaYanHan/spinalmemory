"""The advice path: a question flag and its instruction block, the same text rule for every question.

- Flag: a request for a recommendation, suggestion, tips, ideas or advice, or for an opinion on a choice, by a
  fixed regex over the question as asked (no labels, no model). Never when the "refers to the assistant" rule
  fires: "Can you remind me what you recommended ..." asks to recall a past suggestion.
- Instruction block: asks for one suggestion tailored to the user, in 1-3 sentences, naming the remembered
  details it builds on. Memory block, question and system prompt are unchanged.
- Output: handles written on the Answer line are cut (render.strip_handles), as for every question.
"""

from __future__ import annotations

import re

from spinalmemory.query.parse import ASSISTANT_REF

ADVICE_REQUEST = re.compile(
    r"\brecommend\w*|\bsuggest\w*|\btips?\b|\bideas?\b|\badvice\b|\badvise\b"
    r"|\bwhat should i\b|\bshould i\b|\bwhat do you think\b|\bany thoughts\b"
    r"|\bhelp me (?:choose|decide|pick|find|plan)\b",
    re.I,
)

OUTRO = (
    "This question asks for a recommendation or advice. In the memory, find what the user has said about "
    "themselves that bears on it: their preferences, plans, possessions, routines and past experiences. "
    "Build on what the user already has or does rather than suggesting it again. Reply in exactly this format:\n"
    "Answer: <one suggestion tailored to the user, 1-3 sentences on one line, naming the specific details from "
    "the memory it builds on (e.g. \"Since you ...\"); the suggestion itself may use general knowledge; "
    "no generic list>\n"
    "Evidence: <the handles of the turns you used, e.g. [3] [7]>"
)


def is_advice(question: str) -> bool:
    return bool(ADVICE_REQUEST.search(question)) and not ASSISTANT_REF.search(question)
