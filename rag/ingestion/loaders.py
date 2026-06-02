from __future__ import annotations

import os
from pathlib import Path
from typing import List

from rag.config.retrieval import SUPPORTED_EXTENSIONS
from rag.ingestion.readers import read_document


def scan_document_files(data_dir: str | Path) -> List[str]:
    data_dir = Path(data_dir)
    files: List[str] = []

    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(str(Path(root) / filename))

    return sorted(files)


def load_document_text(file_path: str | Path) -> str:
    return read_document(file_path)
