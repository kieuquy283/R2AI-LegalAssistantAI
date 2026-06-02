from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
INDEX_DIR = PROJECT_ROOT / os.getenv("INDEX_DIR", "data/indexes/default")
LOG_DIR = PROJECT_ROOT / os.getenv("LOG_DIR", "data/logs")


def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
