"""
extract_relations.py — Cross-document Reference Extractor for Vietnamese Legal RAG

Trích xuất quan hệ dẫn chiếu chéo giữa các văn bản pháp luật Việt Nam từ:
- documents.jsonl (preamble & ending → GUIDED_BY / REPLACES)
- chunks.jsonl (nội dung → REFERS_TO / EXCLUDED_REF)

Output: append vào legal_edges.jsonl (định dạng JSON Lines)

Usage:
    python -m src.ingestion.extract_relations
    python -m src.ingestion.extract_relations --docs data/processed/cleaned_documents_enriched.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────────────────────

# Số hiệu văn bản: 117/2020/NĐ-CP, 59/2020/QH14, 36/2024/TT-BTC, v.v.
DOC_NUMBER_RE = re.compile(r"\b\d+/\d{4}/[A-ZĐ0-9\-]+\b")

# GUIDED_BY keywords trong preamble (Căn cứ, Căn cứ Luật, Căn cứ Nghị định)
GUIDED_BY_RE = re.compile(
    r"(?:Căn\s+cứ(?:\s+(?:Luật|Nghị\s+định|Thông\s+tư|Bộ\s+luật))?)",
    re.IGNORECASE | re.UNICODE,
)

# REPLACES keywords ở cuối văn bản (thay thế, bãi bỏ)
# NOTE: "sửa đổi, bổ sung" = AMENDS, không phải REPLACES
REPLACES_RE = re.compile(
    r"(?:thay\s+thế|bãi\s+bỏ|bãi\s+bỏ\s+(?:toàn\s+bộ|một\s+phần))",
    re.IGNORECASE | re.UNICODE,
)

# AMENDS keywords (sửa đổi, bổ sung)
AMENDS_RE = re.compile(
    r"(?:sửa\s+đổi,?\s+bổ\s+sung|sửa\s+đổi|bổ\s+sung)",
    re.IGNORECASE | re.UNICODE,
)

# UI/Website description filter (loại bỏ text không phải nội dung pháp lý)
# NOTE: Be careful not to match real legal content!
UI_TEXT_RE = re.compile(
    r"(?:Nội\s+dung\s+hợp\s+nhất|Văn\s+bản\s+hợp\s+nhất|tính\s+năng\s+phần\s+mềm|"
    r"danh\s+mục\s+văn\s+bản\s+hết\s+hiệu\s+lực|"
    r"CHÍNH\s+SÁCH\s+BẢO\s+VỆ\s+DỮ\s+LIỆU\s+CÁ\s+NHÂN|"
    r"Yêu\s+cầu\s+hỗ\s+trợ|Chú\s+thích\s+màu\s+chỉ\s+dẫn|"
    r"Các\s+nội\s+dung\s+củ��\s+VB\s+này)",
    re.IGNORECASE | re.UNICODE,
)

# Chunk-to-chunk: Điều X ... Nghị định/Luật/Thông tư số YYYY/NNNN/ZZZ
# Nhóm 1 = số điều, Nhóm 2 = số hiệu văn bản
CHUNK_REF_RE = re.compile(
    r"(?:Điều\s+(\d+)).*?"
    r"(?:Nghị\s+định|Luật|Thông\s+tư)\s+số\s+(\d+/\d{4}/[A-ZĐ0-9\-]+)",
    re.IGNORECASE | re.UNICODE,
)

# Negation / exclusion keywords trong cửa sổ ngữ cảnh (~60 ký tự trước match)
NEGATION_RE = re.compile(
    r"(?:không\s+áp\s+dụng|ngoại\s+trừ|trừ\s+trường\s+hợp|không\s+bao\s+gồm)",
    re.IGNORECASE | re.UNICODE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PREAMBLE_WINDOW = 5000   # số ký tự đầu
ENDING_WINDOW = 5000     # số ký tự cuối
CONTEXT_WINDOW_CHARS = 60  # cửa sổ phía trước match để kiểm tra phủ định
CONTEXT_CHARS_BEFORE_MATCH = 600  # tăng lên 600 ký tự để có đủ ngữ cảnh
BATCH_LOG_INTERVAL = 5000  # ghi log tiến độ mỗi N dòng

# Legal document hierarchy (lower number = higher authority)
# Quốc hội > Chính phủ > Bộ/ngành > Địa phương
DOC_LEVEL_MAP = {
    "QH": 1,      # Luật, Bộ luật (Quốc hội)
    "NĐ-CP": 2,   # Nghị định (Chính phủ)
    "TT-": 3,     # Thông tư (Bộ)
    "QĐ-": 4,     # Quyết định (Bộ/tỉnh)
    "CT-": 4,     # Chỉ thị
    "TLT-": 3,    # Thông tư liên tịch
    "VBHN-": 5,   # Văn bản hợp nhất
}

# Regex để xác định cấp văn bản từ số hiệu
DOC_TYPE_RE = re.compile(r"/(QH\d+|NĐ-CP|TT-[A-Z]+|QĐ-[A-Z]+|CT-[A-Z]+|TLT-[A-Z]+|VBHN-[A-Z]+)/")

# Filter "hướng dẫn thi hành" - không phải REPLACES/AMENDS
GUIDANCE_RE = re.compile(
    r"(?:hướng\s+dẫn\s+(?:thi\s+hành|thực\s+hiện|áp\s+dụng|tổ\s+chức\s+thực\s+hiện))",
    re.IGNORECASE | re.UNICODE,
)

# Filter "liên quan đến" generic
GENERIC_REF_RE = re.compile(
    r"(?:liên\s+quan\s+(?:đến|tới)|có\s+liên\s+quan|cùng\s+(?:lĩnh\s+vực|ngành))",
    re.IGNORECASE | re.UNICODE,
)

# Filter markdown link list (UI element)
LINK_LIST_RE = re.compile(
    r"\[.*?\]\(https?://.*?\)",
    re.UNICODE,
)

# Danh sách relation types
REL_GUIDED_BY = "GUIDED_BY"
REL_REPLACES = "REPLACES"
REL_AMENDS = "AMENDS"
REL_REFERS_TO = "REFERS_TO"
REL_EXCLUDED_REF = "EXCLUDED_REF"

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────


def _setup_loggers(base_dir: Path) -> Tuple[logging.Logger, logging.Logger]:
    """
    Tạo 2 logger riêng biệt:
      - pipeline_logger → extractor_pipeline.log (INFO+)
      - error_logger → extractor_errors.log (ERROR+)
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline logger
    pipeline_log_path = base_dir / "extractor_pipeline.log"
    pipeline_logger = logging.getLogger("extractor_pipeline")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.propagate = False

    ph = logging.FileHandler(pipeline_log_path, encoding="utf-8", mode="a")
    ph.setLevel(logging.INFO)
    ph.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    pipeline_logger.handlers = []
    pipeline_logger.addHandler(ph)

    # Error logger
    error_log_path = base_dir / "extractor_errors.log"
    error_logger = logging.getLogger("extractor_errors")
    error_logger.setLevel(logging.ERROR)
    error_logger.propagate = False

    eh = logging.FileHandler(error_log_path, encoding="utf-8", mode="a")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    error_logger.handlers = []
    error_logger.addHandler(eh)

    return pipeline_logger, error_logger


# ──────────────────────────────────────────────────────────────────────────────
# CrossDocReferenceExtractor
# ──────────────────────────────────────────────────────────────────────────────


class CrossDocReferenceExtractor:
    """
    Trích xuất quan hệ dẫn chiếu chéo giữa các văn bản pháp luật Việt Nam.

    Task 1 — Doc-to-Doc:
        - GUIDED_BY: từ preamble (5000 ký tự đầu), sau từ khóa "Căn cứ..."
        - REPLACES: từ ending (5000 ký tự cuối), sau từ khóa "thay thế / bãi bỏ"

    Task 2 — Chunk-to-Chunk:
        - REFERS_TO: từ nội dung chunk, pattern "Điều X ... Nghị định/Luật số Y"
        - EXCLUDED_REF: nếu ngữ cảnh phía trước chứa từ khóa phủ định

    Validation:
        - Tier 1: target_id phải tồn tại trong valid_doc_ids / valid_chunk_ids
        - Tier 2: kiểm tra cửa sổ ngữ cảnh phủ định

    Error handling:
        - pipeline.log: tiến độ
        - errors.log: ngoại lệ với traceback
        - quarantine.jsonl: dòng JSON bị lỗi (pipeline không dừng)
    """

    def __init__(
        self,
        *,
        docs_path: str | Path = "data/processed/documents.jsonl",
        chunks_path: str | Path = "data/processed/chunks.jsonl",
        edges_output_path: str | Path = "data/processed/legal_edges.jsonl",
        quarantine_path: str | Path = "data/processed/quarantine_records.jsonl",
        log_dir: str | Path = "data/processed",
        preamble_window: int = PREAMBLE_WINDOW,
        ending_window: int = ENDING_WINDOW,
        context_window_chars: int = CONTEXT_WINDOW_CHARS,
    ) -> None:
        self.docs_path = Path(docs_path)
        self.chunks_path = Path(chunks_path)
        self.edges_output_path = Path(edges_output_path)
        self.quarantine_path = Path(quarantine_path)
        self.log_dir = Path(log_dir)

        self.preamble_window = preamble_window
        self.ending_window = ending_window
        self.context_window_chars = context_window_chars

        # Logger
        self.pipeline_logger, self.error_logger = _setup_loggers(self.log_dir)

        # Validation sets & lookup dicts (loaded trong run())
        self.valid_doc_ids: Set[str] = set()
        self.valid_chunk_ids: Set[str] = set()
        self.doc_number_to_doc_id: Dict[str, str] = {}
        # Index để tra cứu article chunk_id theo (doc_id, article_num) — O(1)
        self.article_index: Dict[Tuple[str, str], str] = {}

        # Statistics
        self.stats: Dict[str, int] = {
            "docs_scanned": 0,
            "chunks_scanned": 0,
            "guided_by_found": 0,
            "replaces_found": 0,
            "amends_found": 0,
            "refers_to_found": 0,
            "excluded_ref_found": 0,
            "rejected_by_validation": 0,
            "rejected_by_hierarchy": 0,
            "rejected_by_year": 0,
            "rejected_by_guidance": 0,
            "quarantined": 0,
            "edges_written": 0,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: stream JSONL
    # ──────────────────────────────────────────────────────────────────────────

    def _stream_jsonl(
        self, path: Path
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """
        Đọc JSONL theo dòng, trả về (line_number, record).
        Dòng lỗi được ghi vào quarantine mà không dừng pipeline.
        """
        if not path.exists():
            self.error_logger.error("File not found: %s", path)
            raise FileNotFoundError(f"Missing file: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                    yield line_no, record
                except json.JSONDecodeError as exc:
                    self.stats["quarantined"] += 1
                    self._write_quarantine(
                        {
                            "file": str(path),
                            "line_no": line_no,
                            "raw": raw_line,
                            "error": f"JSONDecodeError: {exc}",
                        }
                    )
                    self.pipeline_logger.warning(
                        "Quarantined line %d in %s: %s", line_no, path, exc
                    )
                except Exception as exc:
                    self.stats["quarantined"] += 1
                    self.error_logger.error(
                        "Unexpected error at line %d in %s:\n%s",
                        line_no,
                        path,
                        traceback.format_exc(),
                    )
                    self._write_quarantine(
                        {
                            "file": str(path),
                            "line_no": line_no,
                            "raw": raw_line,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    def _write_quarantine(self, entry: Dict[str, Any]) -> None:
        """Ghi 1 dòng vào quarantine file (append, atomic line write)."""
        self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.quarantine_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _write_edge(self, edge: Dict[str, Any]) -> None:
        """Ghi 1 cạnh vào edges output file (append)."""
        self.edges_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.edges_output_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(edge, ensure_ascii=False) + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Legal hierarchy helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_doc_level(doc_number: str) -> int:
        """Get legal hierarchy level (1=high, 5=low) from doc number suffix."""
        if not doc_number:
            return 99
        match = DOC_TYPE_RE.search(doc_number)
        if match:
            doc_type = match.group(1)
            return DOC_LEVEL_MAP.get(doc_type, 99)
        return 99

    @staticmethod
    def _extract_year(doc_number: str) -> int:
        """Extract year from doc number like '100/2019/NĐ-CP'."""
        match = re.search(r"/(\d{4})/", doc_number)
        if match:
            return int(match.group(1))
        return 0

    def _check_hierarchy_validity(self, source_doc_number: str, target_doc_number: str, relation_type: str) -> bool:
        """Check if relation is legally valid between document levels.
        
        Hierarchy (lower number = higher authority):
        1 = Luật/QH, 2 = Nghị định/NĐ-CP, 3 = Thông tư/TT, 4 = Quyết định/QĐ
        
        Rules:
        - GUIDED_BY: Any level can be guided by higher level (TT guided by ND, ND guided by Luật)
        - AMENDS/REPLACES: Source must be SAME level or HIGHER authority than target
          (Luật can amend Luật, ND can amend ND, but TT cannot amend ND or Luật)
        """
        source_level = self._get_doc_level(source_doc_number)
        target_level = self._get_doc_level(target_doc_number)

        # Skip if either is unknown
        if source_level >= 99 or target_level >= 99:
            return True  # Can't determine, allow

        if relation_type in (REL_REPLACES, REL_AMENDS):
            # Source must have equal or higher authority than target
            # Higher authority = lower level number
            # e.g., source_level=1 (Luật), target_level=2 (ND) → OK (Luật amends ND? No wait...)
            # Actually: ND cannot amend Luật. Only Luật can amend Luật.
            # So source_level must be <= target_level (same or higher authority)
            if source_level > target_level:
                # source has LOWER authority than target → invalid
                # e.g., TT (3) trying to amend ND (2) → invalid
                return False
        
        return True

    def _extract_year_from_text(self, text: str) -> int:
        """Extract year from document text/title as fallback."""
        # Look for year patterns in text
        match = re.search(r"năm\s+(\d{4})", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d{4})\s*,?\s*số\s+\d+", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def _check_year_validity(self, source_doc_number: str, target_doc_number: str, relation_type: str, source_text: str = "", target_text: str = "") -> bool:
        """Check if year relationship makes sense.
        
        For AMENDS/REPLACES: source must be same year or later than target.
        A 1997 law cannot amend a 2003 law.
        """
        source_year = self._extract_year(source_doc_number)
        target_year = self._extract_year(target_doc_number)
        
        # Fallback: extract from text
        if source_year == 0 and source_text:
            source_year = self._extract_year_from_text(source_text)
        if target_year == 0 and target_text:
            target_year = self._extract_year_from_text(target_text)

        if source_year == 0 or target_year == 0:
            return True  # Can't determine

        if relation_type in (REL_REPLACES, REL_AMENDS):
            # Source should be same year or later than target
            # (can't replace/amend a future document)
            if source_year < target_year:
                return False
        
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Task 1: Doc-to-Doc
    # ──────────────────────────────────────────────────────────────────────────

    def _is_ui_description(self, text: str) -> bool:
        """Check if text is UI/website description, not legal content.
        Only reject if multiple UI patterns found (to avoid false positives)."""
        matches = len(UI_TEXT_RE.findall(text))
        return matches >= 2  # 2+ UI patterns = likely UI text
    
    def _is_link_list(self, text: str) -> bool:
        """Check if text is markdown link list (UI element)."""
        link_count = len(LINK_LIST_RE.findall(text))
        return link_count >= 3  # 3+ links = likely a link list
    
    def _is_footer_or_template(self, text: str, source_doc_number: str) -> bool:
        """Check if text is footer/template by looking for multiple markdown links 
        without source document number appearing in the text."""
        link_count = len(LINK_LIST_RE.findall(text))
        if link_count >= 3:
            # Many links = likely footer, but verify source doc is mentioned somewhere
            if source_doc_number and source_doc_number not in text:
                return True  # Footer/template with unrelated doc numbers
        return False

    def _validate_relation_by_keywords(self, context: str, proposed_relation: str, source_doc_number: str, target_doc_number: str) -> Tuple[bool, str, str]:
        """Validate and potentially correct relation based on context keywords.
        
        Returns: (is_valid, corrected_relation, reason)
        """
        context_lower = context.lower()
        
        # ── 1. Self-amend check ──
        # Skip if source and target are the same law with different years
        # e.g., Luật Doanh nghiệp 1997 vs Luật Doanh nghiệp 2003 → these are sequential, not amendments
        if source_doc_number and target_doc_number:
            src_base = re.sub(r"/\d{4}/", "", source_doc_number)
            tgt_base = re.sub(r"/\d{4}/", "", target_doc_number)
            if src_base == tgt_base and source_doc_number != target_doc_number:
                return False, REL_REPLACES, "same_law_different_year_is_replacement_not_amendment"
        
        # ── 2. Multi-doc penalty ──
        # If context mentions >2 document numbers, confidence drops significantly
        doc_nums_in_ctx = set(DOC_NUMBER_RE.findall(context))
        if len(doc_nums_in_ctx) > 2:
            # Multiple docs in context = ambiguous, likely a reference list
            return False, proposed_relation, "multiple_docs_in_context_ambiguous"
        
        # ── 3. Strong GUIDED_BY indicators ──
        guided_by_keywords = [
            r"căn\s+cứ\s+vào",
            r"phù\s+hợp\s+với",
            r"hướng\s+dẫn\s+thi\s+hành",
            r"hướng\s+dẫn\s+thực\s+hiện",
            r"hướng\s+dẫn\s+một\s+số\s+nội\s+dung",  # TT hướng dẫn ND
            r"theo\s+quy\s+định\s+tại",
        ]
        
        # ── 4. Strong AMENDS indicators ──
        amends_keywords = [
            r"sửa\s+đổi,\s*bổ\s+sung",
            r"sửa\s+đổi",
            r"bổ\s+sung",
        ]
        
        # ── 5. Strong REPLACES indicators ──
        replaces_keywords = [
            r"thay\s+thế\s+toàn\s+bộ",
            r"thay\s+thế",
            r"bãi\s+bỏ\s+toàn\s+bộ",
            r"bãi\s+bỏ",
        ]
        
        # ── 6. Check for keyword-relation mismatches ──
        has_guided = any(re.search(kw, context_lower) for kw in guided_by_keywords)
        has_amends = any(re.search(kw, context_lower) for kw in amends_keywords)
        has_replaces = any(re.search(kw, context_lower) for kw in replaces_keywords)
        
        # If proposing AMENDS/REPLACES but context only shows GUIDED_BY keywords
        if proposed_relation in (REL_AMENDS, REL_REPLACES) and has_guided and not has_amends and not has_replaces:
            return False, REL_GUIDED_BY, "context_shows_guided_by_not_amends"
        
        # If proposing GUIDED_BY but context shows AMENDS keywords
        if proposed_relation == REL_GUIDED_BY and has_amends and not has_guided:
            return False, REL_AMENDS, "context_shows_amends_not_guided"
        
        # If proposing AMENDS but context shows REPLACES keywords
        if proposed_relation == REL_AMENDS and has_replaces and not has_amends:
            return False, REL_REPLACES, "context_shows_replaces_not_amends"
        
        # ── 7. Phrase proximity check ──
        # Target doc number must appear within 200 chars of a relation keyword
        # This prevents matching doc numbers from unrelated footer/link lists
        if target_doc_number:
            target_pos = context.find(target_doc_number)
            if target_pos >= 0:
                # Find nearest keyword position
                keyword_positions = []
                for kw_list, _ in [(guided_by_keywords, "guided"), (amends_keywords, "amends"), (replaces_keywords, "replaces")]:
                    for kw in kw_list:
                        match = re.search(kw, context_lower)
                        if match:
                            keyword_positions.append(match.start())
                
                if keyword_positions:
                    nearest_dist = min(abs(target_pos - kp) for kp in keyword_positions)
                    if nearest_dist > 300:
                        # Target doc number too far from any relation keyword
                        return False, proposed_relation, "target_too_far_from_relation_keyword"
        
        return True, proposed_relation, ""

    def _contains_source_doc_number(self, text: str, doc_number: str) -> bool:
        """Check if source document number appears in context text."""
        if not doc_number:
            return False
        return doc_number in text

    def _extract_doc_to_doc(self, record: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Trích xuất GUIDED_BY (preamble) và REPLACES/AMENDS (ending) từ 1 document.
        Yields edge dicts (chưa qua validation tier 1).
        """
        text = record.get("cleaned_text") or ""
        if not text:
            return

        # Skip UI description text
        if self._is_ui_description(text):
            return

        doc_id = record.get("doc_id") or record.get("id")
        if not doc_id:
            return

        source_doc_number = record.get("doc_number", "")

        # Clean text: remove "Chú thích màu chỉ dẫn" section and everything after it
        # This is LuatVietnam UI element that appears at the end of documents
        annotation_marker = "Chú thích màu chỉ dẫn"
        if annotation_marker in text:
            text = text[:text.index(annotation_marker)]
        
        # Also remove markdown link lists (UI elements)
        text = LINK_LIST_RE.sub("", text)

        # ---- Preamble: GUIDED_BY ----
        preamble = text[: self.preamble_window]
        for match in DOC_NUMBER_RE.finditer(preamble):
            doc_number = match.group(0)
            # Tăng context window lên 600 ký tự
            start_pos = max(0, match.start() - CONTEXT_CHARS_BEFORE_MATCH)
            context = preamble[start_pos : match.start()]
            if GUIDED_BY_RE.search(context):
                yield {
                    "source_id": str(doc_id),
                    "source_doc_number": source_doc_number,
                    "target_doc_number": doc_number,
                    "relation_type": REL_GUIDED_BY,
                    "confidence": 1.0,
                    "ref_text": doc_number,
                    "context_region": "preamble",
                }

        # ---- Ending: REPLACES or AMENDS ----
        ending = text[-self.ending_window :] if len(text) > self.ending_window else text
        # Skip if ending is footer/template with multiple links
        if self._is_footer_or_template(ending, source_doc_number):
            return  # Skip entire ending
        for match in DOC_NUMBER_RE.finditer(ending):
            doc_number = match.group(0)
            # Tăng context window lên 600 ký tự
            start_pos = max(0, match.start() - CONTEXT_CHARS_BEFORE_MATCH)
            context = ending[start_pos : match.start()]
            
            # Skip if context is about "hướng dẫn thi hành"
            if GUIDANCE_RE.search(context):
                continue
            
            # Skip generic references
            if GENERIC_REF_RE.search(context):
                continue
            
            if REPLACES_RE.search(context):
                # Validate hierarchy and year
                if not self._check_hierarchy_validity(source_doc_number, doc_number, REL_REPLACES):
                    continue
                if not self._check_year_validity(source_doc_number, doc_number, REL_REPLACES, text):
                    continue
                # Validate by keywords
                is_valid, corrected_relation, reason = self._validate_relation_by_keywords(
                    context, REL_REPLACES, source_doc_number, doc_number
                )
                if not is_valid:
                    # Relation mismatch detected by keywords
                    pass  # Will yield corrected relation below
                yield {
                    "source_id": str(doc_id),
                    "source_doc_number": source_doc_number,
                    "target_doc_number": doc_number,
                    "relation_type": corrected_relation if not is_valid else REL_REPLACES,
                    "confidence": 0.9 if not is_valid else 1.0,
                    "ref_text": doc_number,
                    "context_region": "ending",
                    "validation_note": reason if reason else None,
                }
            elif AMENDS_RE.search(context):
                # Validate hierarchy and year
                if not self._check_hierarchy_validity(source_doc_number, doc_number, REL_AMENDS):
                    continue
                if not self._check_year_validity(source_doc_number, doc_number, REL_AMENDS, text):
                    continue
                # Validate by keywords
                is_valid, corrected_relation, reason = self._validate_relation_by_keywords(
                    context, REL_AMENDS, source_doc_number, doc_number
                )
                yield {
                    "source_id": str(doc_id),
                    "source_doc_number": source_doc_number,
                    "target_doc_number": doc_number,
                    "relation_type": corrected_relation if not is_valid else REL_AMENDS,
                    "confidence": 0.9 if not is_valid else 1.0,
                    "ref_text": doc_number,
                    "context_region": "ending",
                    "validation_note": reason if reason else None,
                }

    # ──────────────────────────────────────────────────────────────────────────
    # Task 2: Chunk-to-Chunk
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_chunk_to_chunk(self, record: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Trích xuất REFERS_TO / EXCLUDED_REF từ nội dung 1 chunk.
        Yields edge dicts (chưa qua validation tier 1).
        """
        content = record.get("content") or ""
        if not content:
            return

        chunk_id = record.get("chunk_id") or record.get("id")
        source_doc_id = record.get("doc_id")
        if not chunk_id:
            return

        for match in CHUNK_REF_RE.finditer(content):
            article_num = match.group(1)
            doc_number = match.group(2)
            if not article_num or not doc_number:
                continue

            # Tier 2: kiểm tra cửa sổ ngữ cảnh phủ định
            start_ctx = max(0, match.start() - self.context_window_chars)
            context_window = content[start_ctx : match.start()]

            if NEGATION_RE.search(context_window):
                relation_type = REL_EXCLUDED_REF
                confidence = 0.8
            else:
                relation_type = REL_REFERS_TO
                confidence = 1.0

            yield {
                "source_id": str(chunk_id),
                "source_doc_id": str(source_doc_id) if source_doc_id else None,
                "target_doc_number": doc_number,
                "target_article": article_num,
                "relation_type": relation_type,
                "confidence": confidence,
                "ref_text": match.group(0),
                "context_window": context_window,
            }

    # ──────────────────────────────────────────────────────────────────────────
    # Validation Layer
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_target(self, edge: Dict[str, Any]) -> Optional[str]:
        """
        Tier 1: Xác thực thực thể.
        - Doc-to-Doc: lookup doc_number → doc_id, kiểm tra tồn tại
        - Chunk-to-Chunk: lookup doc_number → doc_id, sau đó tra article_index
          theo (doc_id, article_num) để lấy chunk_id thực tế.
        - Skip self-reference (source == target)
        Trả về resolved target_id hoặc None nếu không hợp lệ.
        """
        doc_number = edge.get("target_doc_number")
        if not doc_number:
            return None

        target_doc_id = self.doc_number_to_doc_id.get(doc_number)
        if not target_doc_id:
            return None

        # Kiểm tra target_doc_id tồn tại
        if target_doc_id not in self.valid_doc_ids:
            return None

        # Skip self-reference
        source_id = edge.get("source_id", "")
        if source_id == target_doc_id:
            return None

        # Doc-to-Doc: target là doc_id
        if edge.get("context_region") in ("preamble", "ending"):
            return target_doc_id

        # Chunk-to-Chunk: tra article_index để lấy chunk_id thực tế
        article_num = edge.get("target_article")
        if article_num is not None:
            key = (target_doc_id, str(article_num))
            target_chunk_id = self.article_index.get(key)
            if target_chunk_id and target_chunk_id in self.valid_chunk_ids:
                # Skip self-reference for chunk-to-chunk too
                if source_id == target_chunk_id:
                    return None
                return target_chunk_id
            # Fallback: dựng ID theo quy ước phổ biến nếu không có trong index
            fallback_id = f"{target_doc_id}_article_{article_num}"
            if fallback_id in self.valid_chunk_ids:
                if source_id == fallback_id:
                    return None
                return fallback_id
            fallback_id2 = f"{target_doc_id}_dieu_{article_num}_article"
            if fallback_id2 in self.valid_chunk_ids:
                if source_id == fallback_id2:
                    return None
                return fallback_id2
            return None

        return target_doc_id

    # ──────────────────────────────────────────────────────────────────────────
    # Build lookup tables
    # ──────────────────────────────────────────────────────────────────────────

    def _build_lookups(self) -> None:
        """
        Xây dựng O(1) lookup tables trước khi quét:
        - valid_doc_ids
        - valid_chunk_ids
        - doc_number → doc_id
        - (doc_id, article_num) → article_chunk_id
        """
        self.pipeline_logger.info("Building lookup tables...")
        t0 = time.time()

        # Load valid doc IDs + doc_number map
        doc_count = 0
        for line_no, record in self._stream_jsonl(self.docs_path):
            doc_count += 1
            doc_id = record.get("doc_id") or record.get("id")
            doc_number = record.get("doc_number")
            if doc_id:
                self.valid_doc_ids.add(str(doc_id))
            if doc_id and doc_number:
                self.doc_number_to_doc_id[str(doc_number)] = str(doc_id)

        # Load valid chunk IDs + article index
        chunk_count = 0
        for line_no, record in self._stream_jsonl(self.chunks_path):
            chunk_count += 1
            chunk_id = record.get("chunk_id") or record.get("id")
            doc_id = record.get("doc_id")
            level = record.get("level")
            article = record.get("article")
            if chunk_id:
                self.valid_chunk_ids.add(str(chunk_id))
            # Index article-level chunks để tra cứu nhanh
            if doc_id and level == "article" and article:
                key = (str(doc_id), str(article))
                self.article_index[key] = str(chunk_id)

        elapsed = time.time() - t0
        self.pipeline_logger.info(
            "Lookup built: %d docs, %d chunks, %d doc_numbers, %d articles in %.2fs",
            doc_count,
            chunk_count,
            len(self.doc_number_to_doc_id),
            len(self.article_index),
            elapsed,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Main runner
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline: build lookups → scan docs → scan chunks → write edges.
        Trả về dict thống kê.
        """
        start_time = time.time()
        self.pipeline_logger.info("=" * 60)
        self.pipeline_logger.info("CrossDocReferenceExtractor started")
        self.pipeline_logger.info("docs_path=%s", self.docs_path)
        self.pipeline_logger.info("chunks_path=%s", self.chunks_path)
        self.pipeline_logger.info("edges_output=%s", self.edges_output_path)

        try:
            # Phase 1: Build lookups
            self._build_lookups()

            # Phase 2: Scan documents (Task 1)
            self.pipeline_logger.info("Phase 2: Scanning documents for doc-to-doc relations...")
            doc_edges = 0
            processed = 0
            for line_no, record in self._stream_jsonl(self.docs_path):
                processed += 1
                self.stats["docs_scanned"] += 1

                for edge in self._extract_doc_to_doc(record):
                    target_id = self._resolve_target(edge)
                    if target_id:
                        final_edge = {
                            "source_id": edge["source_id"],
                            "target_id": target_id,
                            "relation_type": edge["relation_type"],
                            "confidence": edge["confidence"],
                        }
                        self._write_edge(final_edge)
                        self.stats["edges_written"] += 1
                        doc_edges += 1

                        if edge["relation_type"] == REL_GUIDED_BY:
                            self.stats["guided_by_found"] += 1
                        elif edge["relation_type"] == REL_REPLACES:
                            self.stats["replaces_found"] += 1
                        elif edge["relation_type"] == REL_AMENDS:
                            self.stats["amends_found"] += 1
                    else:
                        self.stats["rejected_by_validation"] += 1

                if processed % BATCH_LOG_INTERVAL == 0:
                    self.pipeline_logger.info(
                        "  Docs processed: %d | edges so far: %d",
                        processed,
                        doc_edges,
                    )

            self.pipeline_logger.info(
                "Phase 2 complete. Docs scanned=%d, doc_edges=%d",
                processed,
                doc_edges,
            )

            # Phase 3: Scan chunks (Task 2)
            self.pipeline_logger.info("Phase 3: Scanning chunks for chunk-to-chunk relations...")
            chunk_edges = 0
            processed = 0
            for line_no, record in self._stream_jsonl(self.chunks_path):
                processed += 1
                self.stats["chunks_scanned"] += 1

                for edge in self._extract_chunk_to_chunk(record):
                    target_id = self._resolve_target(edge)
                    if target_id:
                        final_edge = {
                            "source_id": edge["source_id"],
                            "target_id": target_id,
                            "relation_type": edge["relation_type"],
                            "confidence": edge["confidence"],
                        }
                        self._write_edge(final_edge)
                        self.stats["edges_written"] += 1
                        chunk_edges += 1

                        if edge["relation_type"] == REL_REFERS_TO:
                            self.stats["refers_to_found"] += 1
                        elif edge["relation_type"] == REL_EXCLUDED_REF:
                            self.stats["excluded_ref_found"] += 1
                    else:
                        self.stats["rejected_by_validation"] += 1

                if processed % BATCH_LOG_INTERVAL == 0:
                    self.pipeline_logger.info(
                        "  Chunks processed: %d | edges so far: %d",
                        processed,
                        chunk_edges,
                    )

            self.pipeline_logger.info(
                "Phase 3 complete. Chunks scanned=%d, chunk_edges=%d",
                processed,
                chunk_edges,
            )

        except Exception as exc:
            self.error_logger.error(
                "Pipeline crashed:\n%s", traceback.format_exc()
            )
            raise

        elapsed = time.time() - start_time
        self.pipeline_logger.info("=" * 60)
        self.pipeline_logger.info("Pipeline finished in %.2fs", elapsed)
        self.pipeline_logger.info("Stats: %s", json.dumps(self.stats, ensure_ascii=False))
        self.pipeline_logger.info("=" * 60)

        return {
            "elapsed_seconds": elapsed,
            **self.stats,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract cross-document references from Vietnamese legal corpus"
    )
    parser.add_argument(
        "--docs",
        default="data/processed/cleaned_documents_enriched.jsonl",
        help="Path to documents JSONL with cleaned_text (default: data/processed/cleaned_documents_enriched.jsonl)",
    )
    parser.add_argument(
        "--chunks",
        default="data/processed/chunks.jsonl",
        help="Path to chunks.jsonl (default: data/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--edges-out",
        default="data/processed/legal_edges.jsonl",
        help="Output path for edges (append mode)",
    )
    parser.add_argument(
        "--quarantine",
        default="data/processed/quarantine_records.jsonl",
        help="Quarantine file for bad records",
    )
    parser.add_argument(
        "--log-dir",
        default="data/processed",
        help="Directory for extractor_pipeline.log and extractor_errors.log",
    )
    parser.add_argument(
        "--preamble-window",
        type=int,
        default=PREAMBLE_WINDOW,
        help=f"Preamble char window (default: {PREAMBLE_WINDOW})",
    )
    parser.add_argument(
        "--ending-window",
        type=int,
        default=ENDING_WINDOW,
        help=f"Ending char window (default: {ENDING_WINDOW})",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=CONTEXT_WINDOW_CHARS,
        help=f"Negation context window in chars (default: {CONTEXT_WINDOW_CHARS})",
    )
    args = parser.parse_args()

    extractor = CrossDocReferenceExtractor(
        docs_path=args.docs,
        chunks_path=args.chunks,
        edges_output_path=args.edges_out,
        quarantine_path=args.quarantine,
        log_dir=args.log_dir,
        preamble_window=args.preamble_window,
        ending_window=args.ending_window,
        context_window_chars=args.context_chars,
    )

    result = extractor.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
