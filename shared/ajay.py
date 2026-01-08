import re
from typing import Tuple


REPLACEMENTS = {
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\br\b": "are",
    r"\bpls\b": "please",
}


def light_cleanup(text: str) -> str:
    cleaned = text.strip()
    for pattern, replacement in REPLACEMENTS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", cleaned)  # heyyy -> heyy
    cleaned = re.sub(r"!{2,}", "!", cleaned)
    cleaned = re.sub(r"\?{2,}", "?", cleaned)
    cleaned = re.sub(r"([\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]){3,}", r"\1\1", cleaned)
    return cleaned


def ajay_rule(text: str, max_len: int = 90) -> Tuple[str, bool]:
    cleaned = light_cleanup(text)
    if len(cleaned) <= max_len:
        return cleaned, False
    shortened = cleaned[:max_len]
    return shortened, True
