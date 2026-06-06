from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Missing JSONL file: {target}")

    rows: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    target = ensure_parent(path)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, data: Any) -> None:
    target = ensure_parent(path)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    normalized = text or ""
    normalized = normalized.replace("\ufeff", "")
    normalized = re.sub(r"\r\n?", "\n", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


def stable_slug(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^0-9a-zA-Z]+", "_", lowered, flags=re.ASCII)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "item"
