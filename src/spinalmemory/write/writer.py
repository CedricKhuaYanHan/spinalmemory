"""The writer: one LLM call per session window (spinalmemory.llm), at build time only.

The writer is question-, store- and shortlist-blind: it sees one window and nothing else (no session date
either, so it cannot resolve dates; the projection does that in code). It writes 0-3 lines per turn:
a decontextualised sentence, then subject | attribute | value | "raw time phrase".

Outputs are cached per window: key = sha256(prompt version, provider, model, system prompt, exact window
input, max_tokens). A session shared by several histories is written once, and a rebuild never calls the LLM
again.
A window that fails validation is re-ingested once with a reminder line; both attempts are cached.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from spinalmemory import llm
from spinalmemory.paths import WRITER
from spinalmemory.project.vocab import MULTI_VALUED, SINGLE_VALUED
from spinalmemory.write.validate import Validated, validate
from spinalmemory.write.windows import Window

SYSTEM = (
    "You write long-term memory notes for a personal assistant. You read one chat session and write fact lines "
    "that will later be used to answer questions about the people in it. You never answer or continue the "
    "conversation; you only write fact lines in the required format."
)

INSTRUCTIONS = """Write the facts stated in the chat session below, one block per turn that contains facts. Skip turns with nothing worth remembering.

Format (nothing else, no commentary):
^<turn id>
- <sentence> | <subject> | <attribute> | <value> | <time phrase>

Fields:
- sentence: one self-contained sentence stating the fact in the third person, with names instead of pronouns, so it can be understood without the conversation. Keep any time phrase exactly as said (for example "last week"); never turn it into a date.
- subject: who or what the fact is about: the speaker's name (write User for the user), or the name of the other person, pet, place or thing.
- attribute: one of {attributes}; or other:<short_snake_case> if none fits.
- value: the specific value; keep names, titles, numbers and amounts exactly as said.
- time phrase: the words of the turn that say when it happened, copied exactly in double quotes, or - if the turn gives no time.

Rules:
- At most 3 lines per turn, the facts most worth remembering; one fact per line.
- Only facts stated in the turn or its photo; do not guess or add world knowledge.
- Keep preferences, opinions, possessions, relationships, events, plans, counts, places and numbers.
- Skip greetings, thanks and small talk with no specific content.

Example
^x1 Priya: I finally adopted a beagle called Biscuit last weekend! Next month I start my new job at the library.
^x2 Tom: Congrats! I'm still training for the Leeds half marathon, it's in two weeks.
Output:
^x1
- Priya adopted a beagle called Biscuit last weekend. | Priya | pet | Biscuit, a beagle | "last weekend"
- Priya starts a new job at the library next month. | Priya | job | library | "Next month"
^x2
- Tom is training for the Leeds half marathon, which is in two weeks. | Tom | event | Leeds half marathon | "in two weeks"
End of example

Session:
{turns}
Output:
"""

RETRY_NOTE = "\nUse only the turn ids of this session ({first} to {last}), never the example's ids."

PROMPTS = {"w3-v1": (SYSTEM, INSTRUCTIONS)}

_inflight: dict[str, asyncio.Future] = {}


def render_prompt(window: Window, version: str, retry: bool = False) -> tuple[str, str]:
    system, instructions = PROMPTS[version]
    attrs = ", ".join(SINGLE_VALUED + MULTI_VALUED)
    prompt = instructions.replace("{attributes}", attrs).replace("{turns}", window.body)
    if retry and window.ids:
        prompt += RETRY_NOTE.format(first=window.ids[0][0], last=window.ids[-1][0])
    return system, prompt


def cache_key(window: Window, version: str, max_tokens: int, model: llm.Settings) -> str:
    system, prompt = render_prompt(window, version)
    blob = json.dumps({"prompt_version": version, "provider": model.provider, "model": model.model,
                       "system": system, "input": prompt, "max_tokens": max_tokens}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return WRITER / key[:2] / f"{key}.json"


def _cached_record(path: Path) -> dict | None:
    """The cached record for this key, if any."""
    return json.loads(path.read_text()) if path.exists() else None


async def _call(system: str, prompt: str, max_tokens: int, model: llm.Settings) -> dict:
    t0 = time.time()
    text, usage = await llm.chat(prompt, settings=model, system=system, max_tokens=max_tokens)
    return {"text": text, "usage": usage, "seconds": round(time.time() - t0, 3)}


async def _write_uncached(window: Window, version: str, max_tokens: int, max_lines: int,
                          model: llm.Settings) -> dict:
    system, prompt = render_prompt(window, version)
    attempts = [await _call(system, prompt, max_tokens, model)]
    if not validate(window, attempts[0]["text"], max_lines).ok:
        system, prompt = render_prompt(window, version, retry=True)
        attempts.append(await _call(system, prompt, max_tokens, model))
    return {"prompt_version": version, "provider": model.provider, "model": model.model,
            "session_id": window.session_id,
            "window_hash": window.hash, "max_tokens": max_tokens, "attempts": attempts}


async def write_window(window: Window, version: str, max_tokens: int, max_lines: int,
                       model: llm.Settings) -> tuple[Validated, dict, bool]:
    """(validated lines, cache record, cached?). Empty windows (no turns for the writer) make no call.
    `model` is the writer's llm.settings(cfg, "writer")."""
    if not window.ids:
        rec = {"prompt_version": version, "session_id": window.session_id, "window_hash": window.hash, "attempts": []}
        return validate(window, "", max_lines), rec, True
    key = cache_key(window, version, max_tokens, model)
    path = _cache_path(key)
    if key in _inflight:  # the same window, wanted by another build in this process
        rec, _ = await asyncio.shield(_inflight[key])
        return choose(window, rec, max_lines), rec, True
    fut = asyncio.get_running_loop().create_future()
    _inflight[key] = fut
    try:
        rec, cached = await _get_or_write(window, version, max_tokens, max_lines, model, key, path)
        fut.set_result((rec, cached))
    except BaseException as exc:
        fut.set_exception(exc)
        fut.exception()  # mark retrieved, so a failure with no waiters is not reported twice
        raise
    finally:
        _inflight.pop(key, None)
    return choose(window, rec, max_lines), rec, cached


async def _get_or_write(window: Window, version: str, max_tokens: int, max_lines: int, model: llm.Settings,
                        key: str, path: Path) -> tuple[dict, bool]:
    """The cached record, or write it. The window's cross-process lock is claimed only once a request slot is
    free, so a process never holds locks for windows it is not about to send (another process waits on those)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    while True:
        if (rec := _cached_record(path)) is not None:
            return rec, True
        async with _slot():
            if _try_lock(lock):
                try:
                    if (rec := _cached_record(path)) is not None:  # it appeared just before we took the lock
                        return rec, True
                    rec = await _write_uncached(window, version, max_tokens, max_lines, model)
                    rec["key"] = key
                    tmp = path.with_suffix(f".{os.getpid()}.tmp")
                    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
                    tmp.replace(path)
                    return rec, False
                finally:
                    lock.unlink(missing_ok=True)
        await asyncio.sleep(5)  # another process is writing this window


_slots: dict[int, asyncio.Semaphore] = {}


def _slot() -> asyncio.Semaphore:
    """Windows being written at once by this process (one event loop): llm.MAX_IN_FLIGHT."""
    k = id(asyncio.get_running_loop())
    if k not in _slots:
        _slots[k] = asyncio.Semaphore(llm.MAX_IN_FLIGHT)
    return _slots[k]


def _owner_alive(lock: Path) -> bool:
    """A lock file holds its owner's pid; a lock whose owner has exited (killed, crashed) is stale."""
    try:
        pid = int(lock.read_text().strip() or 0)
    except (FileNotFoundError, ValueError):
        return True  # being created right now, or unreadable: treat as held
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _try_lock(lock: Path, stale_s: float = 6 * 3600.0) -> bool:
    """Take the window's lock file (holding our pid). A lock whose owner has died, or that is older than
    `stale_s` (pid reuse), is removed and taken. False if a live process holds it."""
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            pass
        try:
            if _owner_alive(lock) and time.time() - lock.stat().st_mtime <= stale_s:
                return False
            lock.unlink(missing_ok=True)  # its writer died without cleaning up
        except FileNotFoundError:
            pass
    return False


def choose(window: Window, rec: dict, max_lines: int) -> Validated:
    """The validated lines of the attempt to keep: the first that passes, else the one with fewer rejected lines."""
    results = [validate(window, a["text"], max_lines) for a in rec["attempts"]]
    if not results:
        return validate(window, "", max_lines)
    for r in results:
        if r.ok:
            return r
    return min(results, key=lambda r: (r.stats["invalid"] + 100 * r.stats["example_ids"], -len(r.lines)))
