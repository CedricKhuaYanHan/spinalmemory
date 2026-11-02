"""The LLM providers: settings resolve env over role over [llm], each provider sends its own request shape (URL,
auth headers, system prompt) and its usage is normalised to one form. Requests go to an httpx.MockTransport;
nothing leaves the process and every key is fake."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from spinalmemory import llm, tokens
from spinalmemory.config import load_config

ENV = ["SPINE_LLM_PROVIDER", "SPINE_LLM_MODEL", "SPINE_LLM_BASE_URL", "SPINE_LLM_API_KEY", "ANTHROPIC_API_KEY",
       "OPENAI_API_KEY"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


def _send(monkeypatch, s: llm.Settings, reply: dict) -> tuple[httpx.Request, str, dict, dict]:
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=reply)

    monkeypatch.setattr(llm, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def run():
        log = tokens.reader_call_log()
        text, usage = await llm.chat("Where does Tom live?", settings=s, system="Be brief.", max_tokens=64)
        return text, usage, log

    text, usage, log = asyncio.run(run())
    return seen[0], text, usage, log


def test_settings_resolve_env_over_role_over_shared(monkeypatch):
    cfg = load_config("default", {"llm.answer.model": "answer-model"})
    assert llm.settings(cfg, "writer") == llm.Settings("anthropic", "claude-sonnet-5", "https://api.anthropic.com")
    assert llm.settings(cfg, "answer").model == "answer-model"
    monkeypatch.setenv("SPINE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("SPINE_LLM_MODEL", "gpt-test")
    assert llm.settings(cfg, "answer") == llm.Settings("openai", "gpt-test", "https://api.openai.com/v1")
    monkeypatch.setenv("SPINE_LLM_PROVIDER", "openai_compatible")
    with pytest.raises(ValueError, match="base_url"):
        llm.settings(cfg, "answer")


def test_anthropic_request_and_usage(monkeypatch):
    monkeypatch.setenv("SPINE_LLM_API_KEY", "test-key")
    s = llm.settings(load_config(), "answer")
    reply = {"model": s.model, "content": [{"type": "text", "text": " Leeds "}],
             "usage": {"input_tokens": 10, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 2,
                       "output_tokens": 3}}
    req, text, usage, log = _send(monkeypatch, s, reply)
    body = json.loads(req.content)
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "test-key" and req.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in req.headers
    assert body["system"] == "Be brief." and body["messages"] == [{"role": "user", "content": "Where does Tom live?"}]
    assert body["max_tokens"] == 64 and text == "Leeds" and log == {"calls": 1, "sent": 1}
    assert usage == {"models": {s.model: {"calls": 1, "prompt": 16, "cached": 4, "cache_write": 2, "completion": 3}}}


@pytest.mark.parametrize("key", [None, "test-key"])
def test_openai_compatible_request_and_usage(monkeypatch, key):
    monkeypatch.setenv("SPINE_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SPINE_LLM_MODEL", "local-model")
    monkeypatch.setenv("SPINE_LLM_BASE_URL", "http://localhost:8000/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")  # never sent to a compatible server
    if key:
        monkeypatch.setenv("SPINE_LLM_API_KEY", key)
    s = llm.settings(load_config(), "answer")
    reply = {"model": "local-model:latest", "choices": [{"message": {"role": "assistant", "content": "Leeds"}}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 5}}}
    req, text, usage, _ = _send(monkeypatch, s, reply)
    body = json.loads(req.content)
    assert str(req.url) == "http://localhost:8000/v1/chat/completions"
    assert req.headers.get("authorization") == (f"Bearer {key}" if key else None)
    assert "x-api-key" not in req.headers
    assert body["messages"] == [{"role": "system", "content": "Be brief."},
                                {"role": "user", "content": "Where does Tom live?"}]
    assert body["max_tokens"] == 64 and text == "Leeds"
    assert usage == {"models": {"local-model": {"calls": 1, "prompt": 12, "cached": 5, "cache_write": 0,
                                                "completion": 2}}}


def test_missing_key_raises_before_sending(monkeypatch):
    for provider in ("anthropic", "openai"):
        s = llm.Settings(provider, "m", llm.DEFAULT_BASE_URL[provider])
        with pytest.raises(RuntimeError, match="API_KEY"):
            asyncio.run(llm.chat("hi", settings=s))
