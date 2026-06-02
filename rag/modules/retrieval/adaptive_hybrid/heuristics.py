from __future__ import annotations

import re
from typing import List


# =========================================================
# Constants
# =========================================================

LEGAL_PATTERNS = [

    "điều",
    "khoản",
    "mục",
    "chương",
    "thông tư",
    "nghị định",
    "luật",
    "quyết định",

    "article",
    "section",
]

FOLLOW_UP_PATTERNS = [

    "vậy",
    "thế",
    "còn",
    "nó",
    "đó",
    "điều đó",
    "việc đó",
    "trường hợp đó",
]

QUESTION_PATTERNS = [

    "là gì",
    "bao nhiêu",
    "ở đâu",
    "khi nào",
    "như thế nào",
    "ra sao",
]

TECHNICAL_PATTERNS = [

    "rag",
    "llm",
    "faiss",
    "bm25",
    "api",
    "gpu",
    "cpu",
    "sql",
    "bert",
]


# =========================================================
# Text Utilities
# =========================================================

def normalize_text(
    text: str
) -> str:
    """
    Normalize text.
    """

    return " ".join(
        str(text)
        .strip()
        .lower()
        .split()
    )


def tokenize(
    text: str
) -> List[str]:
    """
    Lightweight tokenizer.
    """

    text = normalize_text(
        text
    )

    return re.findall(
        r"\w+",
        text,
    )


# =========================================================
# Query Structure Heuristics
# =========================================================

def query_length(
    query: str
) -> int:
    """
    Number of query tokens.
    """

    return len(
        tokenize(query)
    )


def is_short_query(
    query: str,
    threshold: int = 5,
) -> bool:
    """
    Detect short queries.
    """

    return (
        query_length(query)
        <= threshold
    )


def is_long_query(
    query: str,
    threshold: int = 12,
) -> bool:
    """
    Detect long semantic queries.
    """

    return (
        query_length(query)
        >= threshold
    )


# =========================================================
# Numeric / Exact Matching
# =========================================================

def has_numbers(
    query: str
) -> bool:
    """
    Detect numeric references.
    """

    return bool(
        re.search(
            r"\d+",
            query,
        )
    )


def has_exact_patterns(
    query: str
) -> bool:
    """
    Detect legal / exact-match patterns.
    """

    query = normalize_text(
        query
    )

    return any(

        pattern in query

        for pattern
        in LEGAL_PATTERNS
    )


def has_acronym(
    query: str
) -> bool:
    """
    Detect acronyms.

    Example:
        RAG
        BM25
        GPT
    """

    return bool(

        re.search(
            r"\b[A-Z0-9]{2,10}\b",
            query,
        )
    )


# =========================================================
# Conversational Heuristics
# =========================================================

def is_follow_up_query(
    query: str
) -> bool:
    """
    Detect conversational follow-up.
    """

    query = normalize_text(
        query
    )

    return any(

        pattern in query

        for pattern
        in FOLLOW_UP_PATTERNS
    )


def is_question_query(
    query: str
) -> bool:
    """
    Detect explicit question forms.
    """

    query = normalize_text(
        query
    )

    return any(

        pattern in query

        for pattern
        in QUESTION_PATTERNS
    )


# =========================================================
# Semantic / Keyword Heuristics
# =========================================================

def keyword_density(
    query: str
) -> float:
    """
    Estimate keyword density.

    Higher density:
        more sparse-friendly

    Lower density:
        more semantic-friendly
    """

    tokens = tokenize(query)

    if not tokens:
        return 0.0

    unique_tokens = len(
        set(tokens)
    )

    return (
        unique_tokens
        /
        len(tokens)
    )


def has_technical_terms(
    query: str
) -> bool:
    """
    Detect technical terminology.
    """

    query = normalize_text(
        query
    )

    return any(

        term in query

        for term
        in TECHNICAL_PATTERNS
    )


# =========================================================
# Retrieval Preference Heuristics
# =========================================================

def prefer_sparse_retrieval(
    query: str
) -> bool:
    """
    Sparse retrieval is stronger for:
        - exact references
        - legal articles
        - acronyms
        - keyword-heavy queries
    """

    if has_numbers(query):
        return True

    if has_exact_patterns(query):
        return True

    if has_acronym(query):
        return True

    if has_technical_terms(query):
        return True

    if keyword_density(query) > 0.9:
        return True

    return False


def prefer_dense_retrieval(
    query: str
) -> bool:
    """
    Dense retrieval is stronger for:
        - semantic queries
        - long queries
        - conversational queries
    """

    if is_follow_up_query(query):
        return True

    if is_long_query(query):
        return True

    if keyword_density(query) < 0.7:
        return True

    return False


# =========================================================
# Adaptive Retrieval Parameters
# =========================================================

def adaptive_alpha(
    query: str
) -> float:
    """
    Adaptive dense/sparse weighting.

    Returns:
        alpha

    alpha closer to 1:
        dense-heavy

    alpha closer to 0:
        sparse-heavy
    """

    # ================================================
    # Sparse-heavy
    # ================================================

    if prefer_sparse_retrieval(
        query
    ):
        return 0.3

    # ================================================
    # Dense-heavy
    # ================================================

    if prefer_dense_retrieval(
        query
    ):
        return 0.8

    # ================================================
    # Balanced
    # ================================================

    return 0.5


def adaptive_top_k(
    query: str
) -> int:
    """
    Adaptive retrieval depth.
    """

    # ================================================
    # Conversational queries
    # ================================================

    if is_follow_up_query(
        query
    ):
        return 8

    # ================================================
    # Long semantic queries
    # ================================================

    if is_long_query(query):
        return 8

    # ================================================
    # Exact keyword queries
    # ================================================

    if prefer_sparse_retrieval(
        query
    ):
        return 5

    return 6


def adaptive_candidate_k(
    query: str
) -> int:
    """
    Adaptive candidate pool size.
    """

    top_k = adaptive_top_k(
        query
    )

    return max(
        top_k * 3,
        top_k + 10,
    )


# =========================================================
# Retrieval Confidence
# =========================================================

def retrieval_confidence(
    scores: List[float]
) -> float:
    """
    Estimate retrieval confidence.

    Higher:
        retrieval likely reliable

    Lower:
        retrieval uncertain
    """

    if not scores:
        return 0.0

    if len(scores) == 1:
        return scores[0]

    top_score = scores[0]

    second_score = scores[1]

    score_gap = (
        top_score
        - second_score
    )

    confidence = (
        top_score * 0.7
        +
        score_gap * 0.3
    )

    return round(
        confidence,
        4,
    )


# =========================================================
# Retrieval Diagnostics
# =========================================================

def classify_query(
    query: str
) -> str:
    """
    Human-readable query classification.
    """

    if prefer_sparse_retrieval(
        query
    ):
        return "sparse-oriented"

    if prefer_dense_retrieval(
        query
    ):
        return "dense-oriented"

    return "balanced"