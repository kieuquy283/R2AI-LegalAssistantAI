from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MEANINGLESS_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "xin chào",
    "chào",
    "ok",
    "oke",
    "okay",
    "thanks",
    "thank you",
    "cảm ơn",
    "bye",
    "goodbye",
}

FOLLOWUP_KEYWORDS = (
    "còn",
    "vậy",
    "thế",
    "nếu",
    "trường hợp",
    "thì sao",
    "cái đó",
    "việc đó",
    "như vậy",
)

DOMAIN_KEYWORDS = (
    "hải quan",
    "tờ khai",
    "kiểm hóa",
    "chuyển khẩu",
    "nhập kinh doanh",
    "sang tải",
    "pallet",
    "cửa khẩu",
    "thông quan",
)

TECHNICAL_CODE_PATTERN = re.compile(r"\b[A-Z]{1,8}\d{0,4}\b")
NUMBER_PATTERN = re.compile(r"\d+")

ROLE_LABELS = {
    "user": "User",
    "human": "User",
    "assistant": "Assistant",
    "ai": "Assistant",
    "bot": "Assistant",
    "system": "System",
}


def normalize_text(text: str) -> str:
    return text.strip().lower()


def get_turn_content(turn: Mapping[str, Any]) -> str:
    return str(turn.get("content", "")).strip()


def normalize_role(role: Any) -> str:
    normalized = normalize_text(str(role or "user"))
    return ROLE_LABELS.get(normalized, normalized.capitalize() or "User")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    vector_a = np.asarray(a, dtype=float)
    vector_b = np.asarray(b, dtype=float)

    if vector_a.size == 0 or vector_b.size == 0:
        return 0.0

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def is_meaningful_turn(
    turn: Mapping[str, Any],
    min_words: int = 3,
) -> bool:
    raw_content = get_turn_content(turn)
    content = normalize_text(raw_content)

    if not content:
        return False

    if content in MEANINGLESS_PATTERNS:
        return False

    if "?" in raw_content:
        return True

    if NUMBER_PATTERN.search(content):
        return True

    if TECHNICAL_CODE_PATTERN.search(raw_content.upper()):
        return True

    if contains_keyword(content, FOLLOWUP_KEYWORDS):
        return True

    if contains_keyword(content, DOMAIN_KEYWORDS):
        return True

    return len(content.split()) >= min_words


def filter_meaningful_history(
    history: List[Dict[str, Any]],
    min_words: int = 3,
) -> List[Dict[str, Any]]:
    return [
        turn
        for turn in history
        if is_meaningful_turn(turn, min_words=min_words)
    ]


def annotate_history(
    history: List[Dict[str, Any]],
    min_words: int = 3,
) -> List[Tuple[int, Dict[str, Any]]]:
    return [
        (index, turn)
        for index, turn in enumerate(history)
        if is_meaningful_turn(turn, min_words=min_words)
    ]


def compute_recency_score(index: int, total_turns: int) -> float:
    if total_turns <= 0:
        return 0.0

    return float(index + 1) / float(total_turns)


def rank_by_recency(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return history[::-1]


def format_history(
    history: List[Dict[str, Any]],
    max_chars: Optional[int] = None,
) -> str:
    if not history:
        return "No previous conversation."

    lines: List[str] = []
    current_length = 0

    for turn in history:
        line = f"{normalize_role(turn.get('role'))}: {get_turn_content(turn)}"
        line_length = len(line) + (1 if lines else 0)

        if max_chars is not None and max_chars >= 0 and current_length + line_length > max_chars:
            if not lines:
                truncated = line[:max_chars].rstrip()
                return truncated or "No previous conversation."
            break

        lines.append(line)
        current_length += line_length

    return "\n".join(lines)


def extract_query_from_state(state: Mapping[str, Any]) -> str:
    for key in ("rewritten_query", "query", "question"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_selection_metadata(
    strategy: str,
    top_k: int,
    num_input_history: int,
    num_meaningful_history: int,
    num_selected: int,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    recent_window: Optional[int] = None,
    selected_scores: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "strategy": strategy,
        "top_k": int(top_k),
        "alpha": None if alpha is None else float(alpha),
        "beta": None if beta is None else float(beta),
        "recent_window": None if recent_window is None else int(recent_window),
        "num_input_history": int(num_input_history),
        "num_meaningful_history": int(num_meaningful_history),
        "num_selected": int(num_selected),
        "selected_scores": selected_scores or [],
    }
