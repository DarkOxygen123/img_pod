from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PolicyDecision:
    blocked: bool
    reason: Optional[str] = None
    sanitized_text: Optional[str] = None


HARD_BLOCK_KEYWORDS = [
    "child",
    "minor",
    "underage",
    "teen",
    "kid",
    "bestiality",
    "animal sex",
    "rape",
    "non-consensual",
    "forced",
]

EXPLICIT_KEYWORDS = [
    "nude",
    "nudity",
    "sex",
    "blowjob",
    "porn",
    "boobs",
    "breasts",
    "nipples",
    "pussy",
    "penis",
    "vagina",
    "cum",
]


def _contains_any(text: str, words: list[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def evaluate_private_chat(text: str) -> PolicyDecision:
    if _contains_any(text, HARD_BLOCK_KEYWORDS):
        return PolicyDecision(blocked=True, reason="HARD_BLOCK")
    return PolicyDecision(blocked=False)


def sanitize_general(text: str) -> PolicyDecision:
    if _contains_any(text, HARD_BLOCK_KEYWORDS):
        return PolicyDecision(blocked=True, reason="HARD_BLOCK")

    if _contains_any(text, EXPLICIT_KEYWORDS):
        sanitized = (
            text
            + "\n\nSafety constraints: tasteful, non-explicit, fully clothed, no nudity, no explicit sexual acts, waist-up framing."
        )
        return PolicyDecision(blocked=False, sanitized_text=sanitized)

    return PolicyDecision(blocked=False, sanitized_text=text)
