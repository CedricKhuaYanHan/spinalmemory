"""The closed attribute vocabulary (a pinned projection input; its hash goes into the canonical dump header).

SINGLE_VALUED attributes hold one current value per entity at a time: their attribute spines have a head (the
current value) and earlier members. MULTI_VALUED attributes accumulate (a person has many hobbies): their spines
are histories without a head. Every attribute except `alias` has a spine. Anything else the writer must spell
`other:<snake_case>`.
"""

from __future__ import annotations

import hashlib
import json
import re

SINGLE_VALUED = (
    "age", "birthday", "job", "employer", "school", "field_of_study", "degree", "lives_in", "hometown",
    "nationality", "home", "relationship_status", "partner", "car", "phone", "commute", "diet",
    "health_condition", "fitness_routine", "favorite_food", "favorite_drink", "favorite_book",
    "favorite_movie", "favorite_music", "favorite_sport", "favorite_place", "goal", "budget", "schedule",
    "personal_best",
)
MULTI_VALUED = (
    "activity", "hobby", "event", "visited", "attended", "bought", "owns", "pet", "family", "friend",
    "likes", "dislikes", "preference", "opinion", "plans_to", "achievement", "created", "read", "watched",
    "learned", "health", "feeling", "gift", "count", "experience", "work_project", "alias",
)
VOCABULARY = SINGLE_VALUED + MULTI_VALUED
OTHER = re.compile(r"^other:[a-z0-9_]{1,40}$")
VOCAB_HASH = hashlib.sha256(json.dumps([SINGLE_VALUED, MULTI_VALUED]).encode()).hexdigest()[:16]


def valid_attribute(attr: str) -> bool:
    return attr in VOCABULARY or bool(OTHER.match(attr))


def is_single_valued(attr: str) -> bool:
    return attr in SINGLE_VALUED
