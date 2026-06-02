from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .utils import (
    DEFAULT_EMPTY_HISTORY,
    get_turn_content,
    get_turn_role,
    validate_turn,
)


def truncate_history(
    history: List[Dict[str, Any]],
    max_messages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if max_messages is None or max_messages <= 0:
        return history
    return history[-max_messages:]


def format_turn(turn: Mapping[str, Any]) -> str:
    return f"{get_turn_role(turn)}: {get_turn_content(turn)}"


def format_history_for_rewrite(
    history: List[Dict[str, Any]],
    max_messages: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    if not history:
        return DEFAULT_EMPTY_HISTORY

    selected_history = truncate_history(history, max_messages=max_messages)
    lines: List[str] = []
    current_length = 0

    for turn in selected_history:
        if not validate_turn(turn):
            continue

        line = format_turn(turn)
        line_length = len(line) + (1 if lines else 0)

        if max_chars is not None and max_chars >= 0 and current_length + line_length > max_chars:
            if not lines:
                truncated = line[:max_chars].rstrip()
                return truncated or DEFAULT_EMPTY_HISTORY
            break

        lines.append(line)
        current_length += line_length

    if not lines:
        return DEFAULT_EMPTY_HISTORY

    return "\n".join(lines)
