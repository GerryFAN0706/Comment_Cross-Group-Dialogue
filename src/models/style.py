import re
import pandas as pd
from typing import List, Dict
from ..utils.text_utils import has_question, has_enumeration, has_numbers, has_time_markers, char_len, contains_any

def extract_style_features(text: str, empathy_markers: List[str], politeness_markers: List[str], hedges: List[str]) -> Dict[str, float]:
    text = text or ""
    return {
        "len_chars": char_len(text),
        "has_question": int(has_question(text)),
        "has_enumeration": int(has_enumeration(text)),
        "has_numbers": int(has_numbers(text)),
        "has_time_markers": int(has_time_markers(text)),
        "empathy": int(contains_any(text, empathy_markers)),
        "politeness": int(contains_any(text, politeness_markers)),
        "hedges": int(contains_any(text, hedges)),
    }
