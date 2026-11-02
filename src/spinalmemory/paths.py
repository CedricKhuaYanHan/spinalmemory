"""Where spinalmemory keeps its caches and finds its configs.

    SPINE_CACHE     cache root (default ~/.cache/spinalmemory): stores, writer outputs, vectors
    SPINE_CONFIGS   config directory (default the repository's configs/)
"""

from __future__ import annotations

import os
from pathlib import Path

CACHE = Path(os.environ.get("SPINE_CACHE") or Path.home() / ".cache" / "spinalmemory")
STORES = CACHE / "stores"            # one directory per history: Markdown truth files + projected index
WRITER = CACHE / "writer"            # writer outputs, one JSON per session window
VECTORS = CACHE / "vectors.sqlite"   # encoder vectors keyed by sha256(model revision, text)
CONFIGS = Path(os.environ.get("SPINE_CONFIGS") or Path(__file__).resolve().parents[2] / "configs")
