"""The only route to a generative model: one chat request over HTTP (httpx), to one of three providers.

- anthropic: the Messages API, POST {base_url}/v1/messages (default base_url https://api.anthropic.com).
- openai: chat completions, POST {base_url}/chat/completions (default base_url https://api.openai.com/v1).
- openai_compatible: the same chat-completions API on any server that speaks it (vLLM, Ollama, LM Studio,
  llama.cpp, LiteLLM). base_url is required. Without a key no auth header is sent.

Two roles make requests: the writer at build time (one per session) and the answer call at query time (one per
question). `settings(cfg, role)` resolves each role's provider, model and base_url: environment first
(SPINE_LLM_PROVIDER, SPINE_LLM_MODEL, SPINE_LLM_BASE_URL), then the config's [llm.writer] or [llm.answer], then
[llm]. The defaults live in configs/default.toml only. A role section that sets its own provider does not inherit
the [llm] base_url.

The key comes from the environment only and is read at request time: SPINE_LLM_API_KEY, else ANTHROPIC_API_KEY
(anthropic) or OPENAI_API_KEY (openai). openai_compatible reads SPINE_LLM_API_KEY alone, so a provider key never
goes to a third-party server. The key is never logged or stored.

A caller may rebind `chat` (a module attribute) to its own transport with the same signature. Packing counts
tokens with the budget tokenizer in `spinalmemory.tokens`; the API's counts are only reported as usage.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from functools import lru_cache

import httpx

from spinalmemory import tokens

PROVIDERS = ("anthropic", "openai", "openai_compatible")
DEFAULT_BASE_URL = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com/v1"}
KEY_FALLBACK = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
ROLES = ("writer", "answer")
TOKENIZER_MODEL = tokens.TOKENIZER_MODEL
MAX_IN_FLIGHT = int(os.environ.get("SPINE_LLM_IN_FLIGHT", "16"))  # writer windows sent at once per process

count_tokens = tokens.count_tokens
count_prompt_tokens = tokens.count_prompt_tokens
tokenizer = tokens.tokenizer

_clients: dict[int, httpx.AsyncClient] = {}


@dataclass(frozen=True)
class Settings:
    """Where one role's requests go. Holds no key."""

    provider: str
    model: str
    base_url: str

    @property
    def api_key(self) -> str | None:
        fallback = KEY_FALLBACK.get(self.provider)
        return os.environ.get("SPINE_LLM_API_KEY") or (os.environ.get(fallback) if fallback else None) or None


def settings(cfg: dict, role: str) -> Settings:
    """The provider, model and base_url for `role` ("writer" or "answer"): env, then [llm.<role>], then [llm]."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, not {role!r}")
    shared, own = cfg["llm"], cfg["llm"].get(role, {})
    env = os.environ.get
    provider = env("SPINE_LLM_PROVIDER") or own.get("provider") or shared["provider"]
    model = env("SPINE_LLM_MODEL") or own.get("model") or shared["model"]
    base_url = env("SPINE_LLM_BASE_URL") or own.get("base_url") or ("" if own.get("provider") else shared["base_url"])
    if provider not in PROVIDERS:
        raise ValueError(f"llm provider must be one of {PROVIDERS}, not {provider!r}")
    base_url = (base_url or DEFAULT_BASE_URL.get(provider, "")).rstrip("/")
    if not base_url:
        raise ValueError(f"{provider} needs a base_url (SPINE_LLM_BASE_URL or [llm] base_url)")
    if not model:
        raise ValueError("set a model (SPINE_LLM_MODEL or [llm] model)")
    return Settings(provider, model, base_url)


def request(s: Settings, prompt: str, system: str | None, max_tokens: int,
            stop: list[str] | None) -> tuple[str, dict, dict]:
    """(url, headers, JSON body) for one request. Raises if the provider needs a key and none is set."""
    key = s.api_key
    headers = {"content-type": "application/json"}
    if s.provider == "anthropic":
        if not key:
            raise RuntimeError("set SPINE_LLM_API_KEY (or ANTHROPIC_API_KEY) to call the model")
        headers |= {"anthropic-version": "2023-06-01", "x-api-key": key}
        body = {"model": s.model, "max_tokens": max_tokens, "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        if stop:
            body["stop_sequences"] = stop
        return f"{s.base_url}/v1/messages", headers, body
    if s.provider == "openai" and not key:
        raise RuntimeError("set SPINE_LLM_API_KEY (or OPENAI_API_KEY) to call the model")
    if key:
        headers["authorization"] = f"Bearer {key}"
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    cap = "max_completion_tokens" if s.provider == "openai" else "max_tokens"  # local servers know max_tokens
    body = {"model": s.model, "messages": messages, cap: max_tokens}
    if stop:
        body["stop"] = stop
    return f"{s.base_url}/chat/completions", headers, body


def parse(s: Settings, data: dict) -> tuple[str, dict]:
    """(text, usage) from a response. Usage is {"models": {model: {calls, prompt, cached, cache_write,
    completion}}}, where prompt counts every input token, cached ones included."""
    u = data.get("usage") or {}
    if s.provider == "anthropic":
        if data.get("model") != s.model:
            raise RuntimeError(f"expected {s.model}, the API answered with {data.get('model')}")
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        read, write = int(u.get("cache_read_input_tokens") or 0), int(u.get("cache_creation_input_tokens") or 0)
        prompt, completion = int(u.get("input_tokens") or 0) + read + write, int(u.get("output_tokens") or 0)
    else:  # chat completions; the served model name may differ (a dated snapshot, a local tag), so it is not checked
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        read, write = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0), 0
        prompt, completion = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    usage = {"models": {s.model: {"calls": 1, "prompt": prompt, "cached": read, "cache_write": write,
                                  "completion": completion}}}
    return text.strip(), usage


def _client() -> httpx.AsyncClient:
    loop = id(asyncio.get_running_loop())  # an httpx client cannot outlive its event loop
    if loop not in _clients:
        _clients[loop] = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    return _clients[loop]


async def chat(prompt: str, *, settings: Settings, system: str | None = None, max_tokens: int = 512,
               stop: list[str] | None = None) -> tuple[str, dict]:
    """One request. Returns (text, usage). Raises on HTTP errors and never retries. Every request is counted in
    the active `tokens.reader_call_log()`; one that never left this process is taken back off."""
    url, headers, body = request(settings, prompt, system, max_tokens, stop)
    tokens.record_call()
    tokens.record_sent(1)
    try:
        resp = await _client().post(url, json=body, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout):  # never left this process: not a request to the model
        tokens.record_sent(-1)
        raise
    resp.raise_for_status()
    return parse(settings, resp.json())


@lru_cache(maxsize=1)
def reader_tokenizer_id() -> str:
    """The budget tokenizer's model id plus its cached snapshot revision (a pinned projection input)."""
    from pathlib import Path

    from huggingface_hub.constants import HF_HUB_CACHE

    snaps = Path(HF_HUB_CACHE) / ("models--" + TOKENIZER_MODEL.replace("/", "--")) / "snapshots"
    revs = sorted(p.name for p in snaps.iterdir()) if snaps.exists() else []
    return f"{TOKENIZER_MODEL}@{revs[0] if len(revs) == 1 else 'unknown'}"
