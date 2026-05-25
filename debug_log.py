#!/usr/bin/env python3
"""
debug_log.py — a tiny thread-safe in-memory ring buffer for debug output.

Shared sink for the launcher's Debug tab. Every Gemini request/response is logged here
(see analyze_mutants.py), along with any other debug-level statements. Memory is bounded
by an entry cap so long sessions don't grow without limit.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

_MAXLEN = int(os.getenv("DEBUG_LOG_LINES", "4000"))
_ENTRIES: deque[dict] = deque(maxlen=_MAXLEN)
_LOCK = threading.Lock()
_ENTRY_ID = 0


def log(tag: str, body: str = "") -> None:
    """Append a tagged, timestamped entry."""
    global _ENTRY_ID
    ts = time.strftime("%H:%M:%S")
    with _LOCK:
        _ENTRIES.append({"id": _ENTRY_ID, "ts": ts, "tag": tag, "body": body})
        _ENTRY_ID += 1


def entries() -> list[dict]:
    with _LOCK:
        return list(_ENTRIES)


def lines() -> list[str]:
    """Backward-compat flat view (tag + indented body lines)."""
    with _LOCK:
        result = []
        for e in _ENTRIES:
            result.append(f"[{e['ts']}] {e['tag']}")
            for line in e["body"].splitlines():
                result.append(f"    {line}")
        return result


def clear() -> None:
    with _LOCK:
        _ENTRIES.clear()
