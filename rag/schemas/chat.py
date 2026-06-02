from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class ChatTurn:
    role: str
    content: str


@dataclass(slots=True)
class ChatResult:
    answer: str
    rewritten_query: str
    used_rewrite: bool
    mode: str
    warning: str | None = None
    top_files: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_docs_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
