"""Legal answer generation lives here."""

from legal_rag.generation.answer_generator import LegalAnswerGenerator
from legal_rag.generation.citation_validator import CitationValidationResult, ensure_citations, validate_citations
from legal_rag.generation.quality_checker import AnswerQualityReport, check_answer_quality

__all__ = [
    "AnswerQualityReport",
    "CitationValidationResult",
    "LegalAnswerGenerator",
    "check_answer_quality",
    "ensure_citations",
    "validate_citations",
]
