# Evaluation

## Article-level metrics

Implemented in `src/legal_rag/evaluation/metrics.py`:

- Precision
- Recall
- F2
- Macro averages across questions

## Evaluator

`src/legal_rag/evaluation/evaluator.py` compares:

- prediction file: `results.json`
- gold file: local JSON with `id` and `relevant_articles`

Output format:

```json
{
  "macro_precision": 0.0,
  "macro_recall": 0.0,
  "macro_f2": 0.0,
  "num_questions": 0,
  "details": []
}
```

## CLI

```bash
python scripts/evaluate_submission.py \
  --pred data/submissions/sample_results.json \
  --gold data/processed/sample_gold.json \
  --output data/submissions/sample_eval_report.json
```
