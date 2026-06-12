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

## R2AI Stage 1 flow

```bash
python -m src.evaluation.prepare_r2ai_dataset \
  --input data/evaluation/R2AIStage1DATA.json \
  --output data/evaluation/r2ai_stage1_questions.jsonl

python scripts/generate_submission.py \
  --input data/evaluation/r2ai_stage1_questions.jsonl \
  --output data/submissions/results.json
```

## Stage 1 Orchestration

```bash
python scripts/run_r2ai_stage1_pipeline.py --derive-gold
```

If you have a real gold file for Stage 1, place it at `data/processed/r2ai_stage1_gold.json` or pass `--gold` to the orchestration script. The derived gold used in this repo is only a smoke-test placeholder.
