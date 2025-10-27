import re
from typing import List

def has_question(text: str) -> bool:
    return bool(re.search(r"[?？]", text or ""))

def has_hashtag(text: str) -> bool:
    # Weibo hashtags like #topic#
    return bool(re.search(r"#.+?#", text or ""))

def char_len(text: str) -> int:
    return len(text or "")

def contains_any(text: str, phrases: List[str]) -> bool:
    if not text:
        return False
    return any(p in text for p in phrases)

def count_any(text: str, phrases: List[str]) -> int:
    if not text:
        return 0
    return sum(text.count(p) for p in phrases)

def has_enumeration(text: str) -> bool:
    # crude heuristic for 1/2/3 or numbered bullets
    return bool(re.search(r"(^|\s)([（(]?\d+[)）.、])", text or ""))

def has_numbers(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))

def has_time_markers(text: str) -> bool:
    return bool(re.search(r"\d{4}[-/年]\d{1,2}([-/月]\d{1,2})?", text or ""))
