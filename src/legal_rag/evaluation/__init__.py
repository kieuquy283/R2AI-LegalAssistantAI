"""Evaluation helpers for legal retrieval and submission."""

from legal_rag.evaluation.evaluator import evaluate_predictions, evaluate_submission_files
from legal_rag.evaluation.metrics import f2_score, macro_average, precision, recall

__all__ = [
    "evaluate_predictions",
    "evaluate_submission_files",
    "f2_score",
    "macro_average",
    "precision",
    "recall",
]
