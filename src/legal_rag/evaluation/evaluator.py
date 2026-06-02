from __future__ import annotations

from pathlib import Path
from typing import Any

from legal_rag.evaluation.metrics import f2_score, macro_average, precision, recall
from legal_rag.submission.schema import SubmissionItem
from legal_rag.utils import load_json, save_json


def evaluate_predictions(pred_payload: list[dict[str, Any]], gold_payload: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = {int(item["id"]): SubmissionItem.model_validate(item) for item in pred_payload}
    gold_by_id = {int(item["id"]): item for item in gold_payload}

    details: list[dict[str, Any]] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f2_scores: list[float] = []

    for question_id, gold_item in gold_by_id.items():
        prediction = predictions.get(question_id)
        pred_articles = set(prediction.relevant_articles if prediction else [])
        gold_articles = set(gold_item.get("relevant_articles", []))
        p = precision(pred_articles, gold_articles)
        r = recall(pred_articles, gold_articles)
        f2 = f2_score(p, r)
        precision_scores.append(p)
        recall_scores.append(r)
        f2_scores.append(f2)
        details.append(
            {
                "id": question_id,
                "precision": p,
                "recall": r,
                "f2": f2,
                "predicted_articles": sorted(pred_articles),
                "gold_articles": sorted(gold_articles),
                "correct_articles": sorted(pred_articles & gold_articles),
            }
        )

    return {
        "macro_precision": macro_average(precision_scores),
        "macro_recall": macro_average(recall_scores),
        "macro_f2": macro_average(f2_scores),
        "num_questions": len(gold_by_id),
        "details": details,
    }


def evaluate_submission_files(pred_path: str | Path, gold_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    predictions = load_json(pred_path, [])
    gold = load_json(gold_path, [])
    report = evaluate_predictions(predictions, gold)
    if output_path is not None:
        save_json(output_path, report)
    return report
