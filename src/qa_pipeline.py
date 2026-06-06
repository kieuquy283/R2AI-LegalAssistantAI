from __future__ import annotations

import argparse
import json
import sys

from src.generation.answer_generator import AnswerGenerator
from src.generation.grounding_validator import GroundingValidator
from src.retrieval.retrieval_pipeline import RetrievalPipeline


class LegalQAPipeline:
    def __init__(self) -> None:
        self.retrieval_pipeline = RetrievalPipeline()
        self.answer_generator = AnswerGenerator()
        self.grounding_validator = GroundingValidator()

    def answer(self, question: str) -> dict:
        retrieval_result = self.retrieval_pipeline.run(question)
        generated = self.answer_generator.generate(query=question, retrieval_result=retrieval_result)
        grounding = self.grounding_validator.validate(
            query=question,
            answer=str(generated.get("answer") or ""),
            citations=list(generated.get("citations") or []),
            contexts=list(retrieval_result.get("final_contexts") or []),
        )
        return {
            "question": question,
            "route": retrieval_result["route"],
            "domains": retrieval_result["domains"],
            "answer": generated["answer"],
            "citations": generated["citations"],
            "grounding": grounding,
            "retrieved_chunks": retrieval_result["seed_chunks"],
            "expanded_contexts": retrieval_result["expanded_contexts"],
            "final_contexts": retrieval_result["final_contexts"],
        }


def _cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run full legal QA pipeline for one question.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(json.dumps(LegalQAPipeline().answer(args.question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
