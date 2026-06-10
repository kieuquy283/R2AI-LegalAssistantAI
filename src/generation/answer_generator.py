from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List

from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.retrieval_pipeline import RetrievalPipeline


class AnswerGenerator:
    def __init__(
        self,
        *,
        temperature: float = 0.1,
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.llm_client = llm_client or LLMClient(temperature=temperature)
        self.prompt_builder = prompt_builder or PromptBuilder()

    def _collect_citations(self, contexts: List[Dict[str, object]]) -> List[Dict[str, object]]:
        citations: List[Dict[str, object]] = []
        seen = set()
        for context in contexts:
            metadata = dict(context.get("metadata") or {})
            key = (metadata.get("doc_title"), metadata.get("article"), metadata.get("source_url"))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "doc_title": metadata.get("doc_title"),
                    "article": metadata.get("article"),
                    "source_url": metadata.get("source_url"),
                    "citation": metadata.get("citation"),
                }
            )
        return citations

    def _fallback_answer(self, *, contexts: List[Dict[str, object]], citations: List[Dict[str, object]]) -> str:
        if not contexts:
            return (
                "Chưa đủ thông tin đáng tin cậy trong dữ liệu truy xuất để trả lời dứt điểm câu hỏi này; "
                "cần bổ sung thêm tình tiết thực tế hoặc diễn đạt truy vấn cụ thể hơn để xác định đúng quy định áp dụng."
            )

        lead = contexts[0]
        lead_meta = dict(lead.get("metadata") or {})
        snippets: List[str] = []
        for context in contexts[:2]:
            text = re.sub(r"\s+", " ", str(context.get("content") or "")).strip()
            if text:
                snippets.append(text[:280].rstrip(" ,;:"))
        analysis = " ".join(snippets).strip()
        if not analysis:
            analysis = "Ngữ liệu truy xuất cho thấy có quy định liên quan nhưng chưa đủ dài để trích xuất thêm chi tiết an toàn."

        lead_title = str(lead_meta.get("doc_title") or "").strip()
        if lead_title:
            return (
                f"Theo nội dung truy xuất từ {lead_title}, vấn đề này có thể được hiểu như sau: {analysis}. "
                "Doanh nghiệp nên đối chiếu thêm tình tiết thực tế, thời điểm áp dụng và hồ sơ cụ thể trước khi ra quyết định."
            )
        return (
            f"Theo nội dung truy xuất hiện có, vấn đề này có thể được hiểu như sau: {analysis}. "
            "Doanh nghiệp nên đối chiếu thêm tình tiết thực tế và hồ sơ cụ thể trước khi áp dụng."
        )

    def generate(self, *, query: str, retrieval_result: Dict[str, object], use_llm: bool = True) -> Dict[str, object]:
        contexts = list(retrieval_result.get("final_contexts") or [])
        citations = self._collect_citations(contexts)
        used_context_ids = [str(context["chunk_id"]) for context in contexts]
        prompt = self.prompt_builder.build(
            query=query,
            contexts=contexts,
            route=str(retrieval_result.get("route") or ""),
            domains=list(retrieval_result.get("domains") or []),
        )
        llm_answer = None
        if use_llm:
            llm_answer = self.llm_client.generate(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
                temperature=self.temperature,
            )
        answer = llm_answer or self._fallback_answer(contexts=contexts, citations=citations)
        return {
            "answer": answer,
            "citations": citations,
            "used_context_ids": used_context_ids,
            "generation_mode": "llm" if llm_answer else "template",
            "prompt": prompt,
        }


def _cli() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate grounded answer from retrieval pipeline output.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    retrieval = RetrievalPipeline().run(args.query)
    print(json.dumps(AnswerGenerator().generate(query=args.query, retrieval_result=retrieval), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
