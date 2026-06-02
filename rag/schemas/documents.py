from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class ChunkRecord:
    text: str
    metadata: Dict[str, Any]
