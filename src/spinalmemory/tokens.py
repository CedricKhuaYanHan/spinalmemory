"""Token counting for packing budgets, and a per-question log of answer calls.

Budgets are measured with the Qwen3.6 tokenizer (tokenizer files only; no model weights are loaded), so a budget
of 1,500 or 3,000 tokens means the same amount of memory whichever model answers. `count_tokens` is therefore a
budget unit and not the answering model's own token count.

    n = count_tokens("text")
    log = reader_call_log()     # then every answer call made in this task (and its children) is counted
"""

from __future__ import annotations

import contextvars
from functools import lru_cache

TOKENIZER_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"

# Per question: a dict counting answer calls. Child tasks and threads copy the context, so they share the dict.
_CALL_LOG: contextvars.ContextVar[dict | None] = contextvars.ContextVar("spinalmemory_call_log", default=None)


@lru_cache(maxsize=1)
def tokenizer():
    """The budget tokenizer (downloaded once by Hugging Face, then read from its cache)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def count_tokens(text: str) -> int:
    """Tokens of `text` under the budget tokenizer (no chat template)."""
    return len(tokenizer().encode(text, add_special_tokens=False))


def count_prompt_tokens(prompt: str, system: str | None = None) -> int:
    """Tokens of the chat-templated prompt under the budget tokenizer (thinking off)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    ids = tokenizer().apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    if hasattr(ids, "keys"):  # newer transformers return a BatchEncoding
        ids = ids["input_ids"]
    return len(ids)


def reader_call_log() -> dict:
    """Start counting the answer calls of the current task and everything it starts: {"calls", "sent"}.
    `sent` counts requests that may have reached the API (connection errors are taken back off)."""
    log = {"calls": 0, "sent": 0}
    _CALL_LOG.set(log)
    return log


def record_call() -> None:
    log = _CALL_LOG.get()
    if log is not None:
        log["calls"] += 1


def record_sent(n: int) -> None:
    log = _CALL_LOG.get()
    if log is not None:
        log["sent"] += n
