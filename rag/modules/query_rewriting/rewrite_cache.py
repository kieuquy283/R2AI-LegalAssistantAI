from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .utils import extract_strong_entities_and_codes, normalize_text


@dataclass
class RewriteCacheItem:
    key: str
    query: str
    history_text: str
    rewritten_query: str
    query_type: str
    entities: List[str]
    created_at: float
    hit_count: int = 0


class RewriteCache:
    def __init__(
        self,
        max_size: int = 1000,
        min_entity_overlap: float = 0.5,
    ) -> None:
        self.max_size = max_size
        self.min_entity_overlap = min_entity_overlap
        self.cache: Dict[str, RewriteCacheItem] = {}

    def get(
        self,
        query: str,
        history_text: str,
    ) -> Optional[str]:
        key = self.build_cache_key(query=query, history_text=history_text)
        item = self.cache.get(key)
        if item is None:
            return None
        if not self.validate_cache_hit(query=query, history_text=history_text, item=item):
            return None
        item.hit_count += 1
        return item.rewritten_query

    def set(
        self,
        query: str,
        history_text: str,
        rewritten_query: str,
    ) -> None:
        key = self.build_cache_key(query=query, history_text=history_text)
        self.cache[key] = RewriteCacheItem(
            key=key,
            query=query,
            history_text=history_text,
            rewritten_query=rewritten_query,
            query_type=self.detect_query_type(query),
            entities=self.extract_entities(query + " " + history_text),
            created_at=time.time(),
        )
        self.evict_if_needed()

    def validate_cache_hit(
        self,
        query: str,
        history_text: str,
        item: RewriteCacheItem,
    ) -> bool:
        current_query_type = self.detect_query_type(query)
        if current_query_type != item.query_type:
            return False

        current_entities = self.extract_entities(query + " " + history_text)
        overlap = self.compute_entity_overlap(current_entities, item.entities)
        return overlap >= self.min_entity_overlap

    def build_cache_key(
        self,
        query: str,
        history_text: str,
    ) -> str:
        combined = normalize_text(query) + " || " + normalize_text(history_text)
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    def detect_query_type(self, query: str) -> str:
        normalized_query = normalize_text(query)
        if any(keyword in normalized_query for keyword in ("bao nhiêu", "mức phạt", "giá", "chi phí")):
            return "quantity"
        if any(keyword in normalized_query for keyword in ("khi nào", "bao giờ", "thời gian")):
            return "temporal"
        if any(keyword in normalized_query for keyword in ("ở đâu", "nơi nào", "địa điểm")):
            return "location"
        if any(keyword in normalized_query for keyword in ("như thế nào", "xử lý thế nào", "cách", "hướng dẫn")):
            return "procedural"
        if any(keyword in normalized_query for keyword in ("khác gì", "so sánh", "giống nhau")):
            return "comparison"
        return "factual"

    def extract_entities(self, text: str) -> List[str]:
        entities = set(entity.lower() for entity in extract_strong_entities_and_codes(text))
        return sorted(entities)

    def compute_entity_overlap(
        self,
        entities_a: List[str],
        entities_b: List[str],
    ) -> float:
        set_a = set(entities_a)
        set_b = set(entities_b)
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        overlap = len(set_a & set_b)
        return overlap / max(len(set_a), len(set_b))

    def evict_if_needed(self) -> None:
        if len(self.cache) <= self.max_size:
            return
        oldest_key = min(self.cache, key=lambda key: self.cache[key].created_at)
        del self.cache[oldest_key]

    def stats(self) -> Dict[str, int]:
        return {"cache_size": len(self.cache)}
