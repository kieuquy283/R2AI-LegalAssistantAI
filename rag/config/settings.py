from __future__ import annotations

from pathlib import Path

from rag.config.paths import PROJECT_ROOT, ensure_runtime_dirs

ensure_runtime_dirs()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()