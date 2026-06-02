## Adaptive Modular Pipeline

The adaptive runtime is the production chatbot pipeline.

Goals:
- Keep the internal logic of the stable modules unchanged.
- Start with cheaper retrieval routes.
- Escalate only when retrieval confidence is low.
- Reuse the existing answer generation path after retrieval finishes.

Routing levels:
- `SIMPLE_DENSE`: original question with dense retrieval only.
- `REWRITE_DENSE`: recency-based history selection, query rewrite, dense retrieval.
- `HYBRID`: hybrid retrieval without reranking.
- `HYBRID_RERANK`: hybrid retrieval with reranking.
- `FULL_OPTIMAL`: hybrid history selection, query rewriting, multi-query, hybrid retrieval, reranking.
- `HYDE`: rewrite, HyDE expansion, hybrid retrieval, reranking.

Production rule:
- Routing, thresholds, fallback rules, and confidence scoring live only in `rag/modules/routing/` and `rag/pipelines/adaptive_modular_pipeline.py`.
- Retrieval, rewriting, reranking, and history selection modules remain frozen building blocks.
