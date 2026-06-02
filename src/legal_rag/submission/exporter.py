from __future__ import annotations

from pathlib import Path
from typing import Iterable

from legal_rag.submission.schema import SubmissionItem
from legal_rag.utils import save_json


def export_submission(items: Iterable[SubmissionItem], output_path: str | Path) -> None:
    save_json(output_path, [item.model_dump() for item in items])
