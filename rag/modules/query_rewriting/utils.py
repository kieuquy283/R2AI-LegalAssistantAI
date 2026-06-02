from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_EMPTY_HISTORY = "No previous conversation."

VI_EXPLICIT_FOLLOW_UP_PATTERNS = (
    r"^còn(?: .*?)?(?: thì sao)?\??$",
    r"^vậy(?: .*?)?(?: thì sao)?\??$",
    r"^thế(?: .*?)?(?: thì sao)?\??$",
    r"^nếu\b.*",
    r"^trường hợp\b.*",
    r".*\bthì sao\??$",
    r"^cái đó\b.*",
    r"^việc đó\b.*",
    r"^như vậy\b.*",
)

EN_EXPLICIT_FOLLOW_UP_PATTERNS = (
    r"^what about\b.*",
    r"^how about\b.*",
    r"^and then\b.*",
    r"^in that case\b.*",
    r"^what if\b.*",
)

FOLLOW_UP_KEYWORDS = (
    "còn",
    "vậy",
    "thế",
    "nếu",
    "trường hợp",
    "thì sao",
    "cái đó",
    "việc đó",
    "như vậy",
    "what about",
    "how about",
    "and then",
    "in that case",
    "what if",
)

PRONOUN_REFERENCE_KEYWORDS = (
    "nó",
    "họ",
    "đó",
    "này",
    "việc này",
    "việc đó",
    "trường hợp này",
    "trường hợp đó",
    "cái đó",
    "it",
    "they",
    "this",
    "that",
    "those",
    "these",
)

SHORT_AMBIGUOUS_PATTERNS = (
    r"^phạt bao nhiêu\??$",
    r"^có được không\??$",
    r"^được không\??$",
    r"^xử lý thế nào\??$",
    r"^khi nào\??$",
    r"^bao nhiêu\??$",
    r"^bao giờ\??$",
    r"^ở đâu\??$",
    r"^how much\??$",
    r"^when\??$",
)

STANDALONE_PATTERNS = (
    r".*\blà gì\??$",
    r"^giải thích\b.*",
    r"^định nghĩa\b.*",
    r"^hướng dẫn\b.*",
    r"^trình bày\b.*",
    r"^phân tích\b.*",
    r"^what is\b.*",
    r"^define\b.*",
    r"^explain\b.*",
    r"^how to\b.*",
)

ANSWER_STYLE_PATTERNS = (
    r"^\s*answer\s*:",
    r"^\s*response\s*:",
    r"^\s*rewrite\s*:",
    r"^\s*rewritten query\s*:",
    r"^\s*rewritten standalone query\s*:",
    r"^\s*standalone query\s*:",
    r"^\s*trả lời\s*:",
    r"^\s*giải thích\s*:",
)

STRONG_ENTITY_TERMS = (
    "RAG",
    "FAISS",
    "BM25",
    "HS",
    "CO",
    "CQ",
    "A11",
    "E11",
)

STRONG_ENTITY_PATTERN = re.compile(
    r"\b(?:RAG|FAISS|BM25|HS|CO|CQ|A11|E11|[A-Z]{2,10}\d{0,4})\b"
)
LEGAL_REFERENCE_PATTERN = re.compile(
    r"\b(?:Điều|Nghị định|Thông tư)\s+\d+\b", re.IGNORECASE
)
NUMBER_PATTERN = re.compile(r"\d+")
WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class RewriteDecision:
    should_rewrite: bool
    reason: str
    confidence: float
    query_type: str


@dataclass
class RewriteValidationResult:
    passed: bool
    errors: List[str]


def safe_strip(value: Any) -> str:
    return str(value or "").strip()


def normalize_text(text: str) -> str:
    return " ".join(safe_strip(text).lower().split())


def normalize_whitespace(text: str) -> str:
    return " ".join(safe_strip(text).split())


def contains_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in keywords)


def is_vietnamese_query(text: str) -> bool:
    return bool(re.search(r"[ăâđêôơưáàảãạấầẩẫậéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", text.lower()))


def get_turn_role(turn: Mapping[str, Any]) -> str:
    role = normalize_text(turn.get("role", "user"))
    return {
        "human": "User",
        "user": "User",
        "assistant": "Assistant",
        "ai": "Assistant",
        "system": "System",
    }.get(role, role.capitalize() or "User")


def get_turn_content(turn: Mapping[str, Any]) -> str:
    return normalize_whitespace(turn.get("content", ""))


def validate_turn(turn: Mapping[str, Any]) -> bool:
    return isinstance(turn, Mapping) and bool(get_turn_content(turn))


def is_short_query(query: str, threshold: int = 6) -> bool:
    return len(normalize_whitespace(query).split()) <= threshold


def extract_numbers(text: str) -> List[str]:
    return NUMBER_PATTERN.findall(text)


def extract_strong_entities_and_codes(text: str) -> List[str]:
    entities = set()

    for match in STRONG_ENTITY_PATTERN.findall(text):
        entities.add(match.upper())

    for match in LEGAL_REFERENCE_PATTERN.findall(text):
        entities.add(normalize_whitespace(match).lower())

    if re.search(r"\bhs code\b", text, re.IGNORECASE):
        entities.add("HS CODE")

    return sorted(entities)


def has_strong_entity_or_code(query: str) -> bool:
    normalized_query = normalize_whitespace(query)
    if extract_numbers(normalized_query):
        return True
    return bool(extract_strong_entities_and_codes(normalized_query))


def is_standalone_definition_query(query: str) -> bool:
    normalized_query = normalize_text(query)
    return contains_pattern(normalized_query, STANDALONE_PATTERNS)


def is_explicit_follow_up(query: str) -> bool:
    normalized_query = normalize_text(query)
    patterns = VI_EXPLICIT_FOLLOW_UP_PATTERNS + EN_EXPLICIT_FOLLOW_UP_PATTERNS
    return contains_pattern(normalized_query, patterns)


def has_pronoun_reference_dependency(query: str) -> bool:
    return contains_keyword(query, PRONOUN_REFERENCE_KEYWORDS)


def is_short_ambiguous_query(query: str) -> bool:
    normalized_query = normalize_text(query)
    return contains_pattern(normalized_query, SHORT_AMBIGUOUS_PATTERNS)


def analyze_query_dependency(query: str, has_history: bool) -> RewriteDecision:
    normalized_query = normalize_whitespace(query)
    lowered_query = normalize_text(query)

    if not lowered_query:
        return RewriteDecision(False, "empty_query", 1.0, "empty")

    if not has_history:
        query_type = "standalone" if is_standalone_definition_query(normalized_query) else "no_history"
        return RewriteDecision(False, "no_history", 1.0, query_type)

    if is_standalone_definition_query(normalized_query):
        return RewriteDecision(False, "standalone_definition", 0.95, "standalone")

    if is_explicit_follow_up(normalized_query):
        return RewriteDecision(True, "explicit_follow_up", 0.98, "follow_up")

    if has_pronoun_reference_dependency(normalized_query):
        return RewriteDecision(True, "pronoun_reference", 0.92, "reference")

    if is_short_ambiguous_query(normalized_query) and not has_strong_entity_or_code(normalized_query):
        return RewriteDecision(True, "short_ambiguous_with_history", 0.88, "ambiguous")

    if contains_keyword(
        lowered_query,
        ("trường hợp này", "trường hợp đó", "xử lý như thế nào", "xử lý thế nào", "nếu"),
    ):
        return RewriteDecision(True, "context_dependent_long_query", 0.86, "context_dependent")

    if is_short_query(normalized_query) and not has_strong_entity_or_code(normalized_query):
        return RewriteDecision(True, "short_query_without_entity", 0.7, "ambiguous")

    return RewriteDecision(False, "standalone_query", 0.75, "standalone")


def is_likely_follow_up(query: str) -> bool:
    decision = analyze_query_dependency(query, has_history=True)
    return decision.should_rewrite


def clean_rewritten_query(text: str) -> str:
    if not text:
        return ""

    cleaned = safe_strip(text).replace("```", "").strip()
    cleaned = cleaned.splitlines()[0].strip()

    prefixes = [
        "rewritten standalone query:",
        "standalone query:",
        "rewritten query:",
        "rewrite:",
        "query:",
        "trả lời:",
    ]

    lowered = cleaned.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break

    return cleaned.strip("\"'“”‘’").strip()


def _tokenize_keywords(text: str) -> List[str]:
    return [
        token
        for token in WORD_PATTERN.findall(normalize_text(text))
        if len(token) >= 4 and not token.isdigit()
    ]


def validate_rewrite(
    original_query: str,
    rewritten_query: str,
    decision: RewriteDecision,
    max_rewrite_ratio: float = 2.0,
    max_tokens_multiplier: int = 20,
) -> RewriteValidationResult:
    errors: List[str] = []

    original_query = normalize_whitespace(original_query)
    rewritten_query = normalize_whitespace(rewritten_query)

    if not rewritten_query:
        errors.append("empty_rewrite")
        return RewriteValidationResult(False, errors)

    if len(rewritten_query.split()) <= 1:
        errors.append("rewrite_too_short")

    original_len = max(1, len(original_query.split()))
    rewritten_len = len(rewritten_query.split())
    max_allowed = max(max_tokens_multiplier, int(original_len * max_rewrite_ratio))
    if rewritten_len > max_allowed:
        errors.append("rewrite_too_long")

    lowered = rewritten_query.lower()
    if contains_pattern(lowered, ANSWER_STYLE_PATTERNS):
        errors.append("answer_style_output")

    if rewritten_query.count("?") > 3 or rewritten_query.count("!") > 2:
        errors.append("excessive_punctuation")

    if re.search(r"\b(\w+)(?:\s+\1){2,}\b", lowered):
        errors.append("excessive_repetition")

    original_numbers = extract_numbers(original_query)
    rewritten_numbers = extract_numbers(rewritten_query)
    for number in original_numbers:
        if number not in rewritten_numbers:
            errors.append(f"missing_number:{number}")

    original_entities = extract_strong_entities_and_codes(original_query)
    rewritten_entities_upper = {entity.upper() for entity in extract_strong_entities_and_codes(rewritten_query)}
    for entity in original_entities:
        if entity.upper() not in rewritten_entities_upper:
            errors.append(f"missing_entity:{entity}")

    strict_preservation = decision.query_type == "standalone"
    if strict_preservation:
        original_keywords = set(_tokenize_keywords(original_query))
        rewritten_keywords = set(_tokenize_keywords(rewritten_query))
        if len(original_keywords) >= 2:
            overlap = len(original_keywords & rewritten_keywords) / len(original_keywords)
            if overlap < 0.3:
                errors.append("low_keyword_overlap")

    return RewriteValidationResult(not errors, errors)


def should_skip_rewrite(history: List[Dict[str, Any]]) -> bool:
    return not history
