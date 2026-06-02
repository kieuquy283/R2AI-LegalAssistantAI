from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryFeatures:

    query_length: int

    has_numbers: bool

    has_acronym: bool

    has_exact_pattern: bool

    is_follow_up: bool

    keyword_density: float


class QueryAnalyzer:

    def analyze(
        self,
        query: str,
    ) -> QueryFeatures:

        tokens = query.split()

        query_length = len(tokens)

        has_numbers = bool(
            re.search(r"\d+", query)
        )

        has_acronym = bool(
            re.search(
                r"\b[A-Z]{2,10}\b",
                query,
            )
        )

        exact_patterns = [

            "điều",
            "khoản",
            "mục",
            "section",
            "article",
        ]

        has_exact_pattern = any(

            pattern in query.lower()

            for pattern
            in exact_patterns
        )

        follow_up_patterns = [

            "vậy",
            "thế",
            "nó",
            "đó",
        ]

        is_follow_up = any(

            pattern in query.lower()

            for pattern
            in follow_up_patterns
        )

        unique_tokens = len(
            set(tokens)
        )

        keyword_density = (
            unique_tokens
            /
            max(query_length, 1)
        )

        return QueryFeatures(

            query_length=query_length,

            has_numbers=has_numbers,

            has_acronym=has_acronym,

            has_exact_pattern=(
                has_exact_pattern
            ),

            is_follow_up=(
                is_follow_up
            ),

            keyword_density=(
                keyword_density
            ),
        )