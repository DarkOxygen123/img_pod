from __future__ import annotations

import re
from typing import List

MENTION_RE = re.compile(r"@([\w\.]+)")


def normalize_handle(handle: str) -> str:
    h = handle.strip()
    if not h.startswith("@"):  # enforce @ prefix
        h = "@" + h
    return "@" + h[1:].lower()


def extract_mentions(text: str) -> List[str]:
    handles = []
    for m in MENTION_RE.finditer(text or ""):
        handles.append(normalize_handle("@" + m.group(1)))
    # unique preserve order
    seen = set()
    out: List[str] = []
    for h in handles:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out
