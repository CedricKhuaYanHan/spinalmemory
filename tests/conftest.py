"""Test setup: caches in a temporary directory, Hugging Face offline, and a whitespace stand-in for the budget
tokenizer, so no test downloads or loads a model."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("SPINE_CACHE", tempfile.mkdtemp(prefix="spinalmemory-test-"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(self, messages, **kwargs):
        return [w for m in messages for w in m["content"].split()]


@pytest.fixture(autouse=True)
def _stub_tokenizer(monkeypatch):
    from spinalmemory import tokens

    monkeypatch.setattr(tokens, "tokenizer", lambda: WhitespaceTokenizer())
