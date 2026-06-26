import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import List

_LABELED_PATH = Path("data/processed/labeled_dataset_local.jsonl")

# Từ điển ánh xạ các từ viết tắt/thông dụng sang thuật ngữ đầy đủ trong pháp lý/SME
EXPANSION_MAP = {
    r"\bSME\b": "doanh nghiệp nhỏ và vừa",
    r"\bDN\b": "doanh nghiệp",
    r"\bBHXH\b": "bảo hiểm xã hội",
    r"\bBHYT\b": "bảo hiểm y tế",
    r"\bBHTN\b": "bảo hiểm thất nghiệp",
    r"\bHĐLĐ\b": "hợp đồng lao động",
    r"\bNLĐ\b": "người lao động",
    r"\bNSDLĐ\b": "người sử dụng lao động",
    r"\bGTGT\b": "giá trị gia tăng",
    r"\bTNDN\b": "thuế thu nhập doanh nghiệp",
    r"\bTNCN\b": "thuế thu nhập cá nhân",
    r"\bTNHH\b": "trách nhiệm hữu hạn",
    r"\bCTCP\b": "cổ phần",
    r"\bDNTN\b": "doanh nghiệp tư nhân",
    r"\bFDI\b": "đầu tư nước ngoài doanh nghiệp có vốn đầu tư nước ngoài",
    r"\bASEAN\b": "hiệp hội các quốc gia đông nam á",
    r"\bCPTPP\b": "hiệp định đối tác toàn diện và tiến bộ xuyên thái bình dương",
    r"\bEVFTA\b": "hiệp định thương mại tự do việt nam liên minh châu âu",
    r"\bVAT\b": "giá trị gia tăng",
    r"\bGPLX\b": "giấy phép lái xe",
    r"\bCMND\b": "chứng minh nhân dân",
    r"\bCCCD\b": "căn cước công dân",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ỹà-ỹđĐ]+", text.lower())


class LabeledKeywordIndex:
    def __init__(self, path: Path = _LABELED_PATH) -> None:
        self.entries: List[dict] = []
        self.doc_freq: Counter = Counter()
        self.num_docs = 0
        if path.exists():
            self._build(path)

    def _build(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                question = entry.get("question", "")
                keywords = entry.get("tu_khoa_phap_ly", [])
                if not question:
                    continue
                tokens = set(_tokenize(question))
                entry["_tokens"] = tokens
                # Collect keywords from all relevant fields
                kw_set = set()
                for kw in keywords:
                    if kw.strip():
                        kw_set.add(kw.strip().lower())
                # Add linh_vuc (domain) parts as keywords
                linh_vuc = entry.get("linh_vuc", "")
                if linh_vuc:
                    for part in re.split(r"\s*[–\-–]\s*", linh_vuc):
                        part = part.strip()
                        if part and len(part) > 2:
                            kw_set.add(part.lower())
                # Add loai_hinh_sme (full phrase) as keyword
                loai_hinh = entry.get("loai_hinh_sme", "")
                if loai_hinh and len(loai_hinh.strip()) > 2:
                    kw_set.add(loai_hinh.strip().lower())
                entry["_keywords"] = list(kw_set)
                self.entries.append(entry)
        self.num_docs = len(self.entries)
        # Compute document frequency
        for entry in self.entries:
            for token in entry["_tokens"]:
                self.doc_freq[token] += 1
        print(f"[QueryExpander] Loaded {self.num_docs} labeled entries from {path.name}")

    def find_similar(self, query: str, top_k: int = 2, min_jaccard: float = 0.25) -> List[str]:
        if not self.entries:
            return []
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []
        # Score each entry by normalized token overlap (Jaccard-like)
        scored = []
        for entry in self.entries:
            overlap = len(query_tokens & entry["_tokens"])
            union = len(query_tokens | entry["_tokens"])
            score = overlap / union if union > 0 else 0
            if score >= min_jaccard:
                scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        # Collect unique keywords from top matches, skipping those redundant with query
        seen_kws: set[str] = set()
        result: List[str] = []
        for _, entry in scored[:top_k]:
            for kw in entry["_keywords"]:
                kw_tokens = set(_tokenize(kw))
                if not kw_tokens:
                    continue
                key = " ".join(sorted(kw_tokens))
                if key in seen_kws:
                    continue
                # Skip keyword if every token already appears in the query
                if kw_tokens.issubset(query_tokens):
                    continue
                seen_kws.add(key)
                result.append(kw)
        return result


_index_instance: LabeledKeywordIndex | None = None


def _get_index() -> LabeledKeywordIndex:
    global _index_instance
    if _index_instance is None:
        _index_instance = LabeledKeywordIndex()
    return _index_instance


def expand_query(query: str) -> str:
    """
    Mở rộng câu truy vấn.
    - Nếu R2AI_RETRIEVAL_SKIP_EXPANSION=true: không mở rộng, trả về query gốc.
    - Nếu R2AI_USE_KEYWORD_EXPANSION=true: dùng labeled_dataset_local.jsonl keywords.
    - Nếu không: dùng abbreviation expansion cũ.
    """
    skip = os.getenv("R2AI_RETRIEVAL_SKIP_EXPANSION", "").strip().lower() in {"1", "true", "yes"}
    if skip:
        return query.strip()

    use_keyword = os.getenv("R2AI_USE_KEYWORD_EXPANSION", "").strip().lower() in {"1", "true", "yes"}
    if use_keyword:
        index = _get_index()
        keywords = index.find_similar(query)
        if keywords:
            expanded = f"{query} {' '.join(keywords)}"
            print(f"[QueryExpander] Keyword expansion: '{query}' -> '{expanded}' (keywords: {keywords})")
            return expanded
        return query.strip()

    # Legacy abbreviation expansion
    expanded = query
    for abbr, full_text in EXPANSION_MAP.items():
        if re.search(abbr, query, re.IGNORECASE):
            expanded = re.sub(abbr, f"\\g<0> {full_text}", expanded, flags=re.IGNORECASE)
    return expanded.strip()
