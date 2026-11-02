"""Configuration: configs/default.toml, optionally overlaid by a named config and dotted overrides.

    cfg = load_config("default", {"pack.budget": 2000})

A named config only lists what it changes; everything else comes from default.toml. The `[build]` section and
the writer's provider and model (spinalmemory.llm.settings, environment included) decide the store, so they go
into the build key: configs that differ only in query settings or the answer model share one store, and a config
that changes the build or the writer model gets its own.

Code also decides the store: the writer prompt text, the validator rules, the projection, the date resolver,
the attribute vocabulary and the alias rules. Their versions are fingerprinted, and the fingerprint is pinned per
BUILD_VERSION and writer prompt; build_key() refuses to run when the code no longer matches the pin, so a code
change can never silently reuse stores built by older code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from spinalmemory.paths import CONFIGS

# Bump when build() changes what it writes (truth-file format, writer prompt, projection code), then re-pin
# BUILD_FINGERPRINTS with the value build_key() reports.
BUILD_VERSION = "1"
BUILD_FINGERPRINTS = {"w3-v1": "b20fa8052a3e6c1d"}  # writer prompt version -> code_fingerprint() for this BUILD_VERSION


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    if keys[-1] not in node:
        raise KeyError(f"unknown config key {dotted}")
    node[keys[-1]] = value


def load_config(name: str = "default", overrides: dict[str, Any] | None = None, root: Path = CONFIGS) -> dict:
    cfg = tomllib.loads((root / "default.toml").read_text())
    if name and name != "default":
        path = Path(name) if name.endswith(".toml") else root / f"{name}.toml"
        cfg = _merge(cfg, tomllib.loads(path.read_text()))
        cfg["name"] = path.stem
    for k, v in (overrides or {}).items():
        _set_dotted(cfg, k, v)
    if overrides:
        cfg["name"] = cfg["name"] + "+" + ",".join(f"{k}={v}" for k, v in sorted(overrides.items()))
    if not (cfg["render"]["facts"] is True or cfg["render"]["facts"] is False or cfg["render"]["facts"] == "added"):
        raise ValueError(f"render.facts must be true, false or \"added\", not {cfg['render']['facts']!r}")
    if cfg["pack"]["policy"] not in ("fixed", "adaptive"):
        raise ValueError(f"pack.policy must be 'fixed' or 'adaptive', not {cfg['pack']['policy']!r}")
    return cfg


def code_fingerprint(cfg: dict) -> str:
    """The versions of the code that decides a store, for the config's writer prompt."""
    from spinalmemory.project import entities, projection, timeexpr, vocab
    from spinalmemory.write import validate, writer

    system, instructions = writer.PROMPTS[cfg["build"]["writer_prompt"]]
    parts = {"prompt": [system, instructions, writer.RETRY_NOTE], "validator": validate.VALIDATOR_VERSION,
             "projection": projection.PROJECTION_VERSION, "resolver": timeexpr.RESOLVER_HASH,
             "vocabulary": vocab.VOCAB_HASH, "alias_rules": entities.ALIAS_RULES_VERSION}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def build_key(cfg: dict) -> str:
    """The store's build key: shared by every config with the same [build] section and writer model. Raises if
    the store-defining code has changed since BUILD_FINGERPRINTS was pinned for this BUILD_VERSION."""
    from spinalmemory import llm

    fp = code_fingerprint(cfg)
    pinned = BUILD_FINGERPRINTS.get(cfg["build"]["writer_prompt"])
    if fp != pinned:
        raise RuntimeError(
            f"store-defining code changed (fingerprint {fp}, pinned {pinned}): bump BUILD_VERSION in "
            f"spinalmemory/config.py and pin BUILD_FINGERPRINTS[{cfg['build']['writer_prompt']!r}] = {fp!r}, "
            "so stores built by older code are not reused")
    writer = llm.settings(cfg, "writer")
    decides = {"build": cfg["build"], "writer": {"provider": writer.provider, "model": writer.model}}
    digest = hashlib.sha256(json.dumps(decides, sort_keys=True).encode()).hexdigest()[:8]
    return f"spinalmemory-b{BUILD_VERSION}-{digest}"
