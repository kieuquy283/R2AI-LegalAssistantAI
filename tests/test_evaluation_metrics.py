from legal_rag.evaluation.evaluator import evaluate_predictions
from legal_rag.evaluation.metrics import f2_score, precision, recall


def test_precision_recall_f2_macro_primitives():
    pred = {"a", "b"}
    gold = {"b", "c"}
    p = precision(pred, gold)
    r = recall(pred, gold)
    f2 = f2_score(p, r)

    assert p == 0.5
    assert r == 0.5
    assert round(f2, 4) == 0.5


def test_evaluate_predictions_returns_article_level_details():
    predictions = [
        {
            "id": 1,
            "question": "Q1",
            "answer": "Căn cứ Điều 1 Luật A.",
            "relevant_docs": ["A|Luật A"],
            "relevant_articles": ["A|Luật A|Điều 1"],
        }
    ]
    gold = [
        {
            "id": 1,
            "relevant_articles": ["A|Luật A|Điều 1"],
        }
    ]

    report = evaluate_predictions(predictions, gold)

    assert report["macro_precision"] == 1.0
    assert report["macro_recall"] == 1.0
    assert report["macro_f2"] == 1.0
