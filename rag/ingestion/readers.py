from __future__ import annotations

from pathlib import Path

import docx2txt
from langchain_community.document_loaders import PyPDFLoader

from rag.utils.logging import get_logger


logger = get_logger(__name__)


def read_pdf(file_path: str | Path) -> str:
    file_path = Path(file_path)

    # 1) PyPDFLoader
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
        if text and text.strip():
            logger.info("PyPDFLoader extracted %s chars from %s", len(text), file_path.name)
            return text
    except Exception as exc:
        logger.warning("PyPDFLoader failed for %s: %s", file_path, exc)

    # 2) pdfplumber fallback
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)

        text = "\n\n".join(pages)
        if text and text.strip():
            logger.info("pdfplumber extracted %s chars from %s", len(text), file_path.name)
            return text
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s", file_path, exc)

    # 3) PyMuPDF fallback
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(file_path))
        pages = []
        for page in doc:
            page_text = page.get_text("text") or ""
            if page_text.strip():
                pages.append(page_text)

        text = "\n\n".join(pages)
        if text and text.strip():
            logger.info("PyMuPDF extracted %s chars from %s", len(text), file_path.name)
            return text
    except Exception as exc:
        logger.warning("PyMuPDF failed for %s: %s", file_path, exc)

    logger.warning("Could not extract text from PDF: %s", file_path)
    return ""


def read_docx(file_path: str | Path) -> str:
    return docx2txt.process(str(file_path)) or ""


def read_text_file(file_path: str | Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_document(file_path: str | Path) -> str:
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return read_pdf(file_path)
    if ext == ".docx":
        return read_docx(file_path)
    if ext in {".txt", ".md"}:
        return read_text_file(file_path)

    raise ValueError(f"Unsupported file type: {ext}")

def read_text_file(file_path: str | Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1258"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()