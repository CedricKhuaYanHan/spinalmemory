"""spinalmemory: long-term conversational memory with one LLM call per question.

Write path (spinalmemory.write): history -> Markdown truth files, with one writer call per session.
Projection (spinalmemory.project): truth files -> BM25, vector and person-spine indexes, by code alone.
Query path (spinalmemory.query, spinalmemory.answer): question -> search, fact history, rerank, pack, render ->
`Answerer.recall` (the packed memory, no LLM call) or `Answerer.answer` (recall plus exactly one answer call).
"""

__version__ = "0.1.0"

_EXPORTS = {
    "Answerer": "spinalmemory.answer", "Recall": "spinalmemory.answer", "Passage": "spinalmemory.answer",
    "Fact": "spinalmemory.answer", "OneCallViolation": "spinalmemory.answer",
    "build_store": "spinalmemory.write.build", "StoreHandle": "spinalmemory.write.build",
    "load_config": "spinalmemory.config",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):  # imported on first use, so `import spinalmemory` stays light
    if name not in _EXPORTS:
        raise AttributeError(f"module 'spinalmemory' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)
