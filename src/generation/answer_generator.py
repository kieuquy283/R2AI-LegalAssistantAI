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
        temperature: float = 0.05,
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
                "1. Kết luận ngắn\n"
                "Chưa đủ thông tin đáng tin cậy trong dữ liệu truy xuất để trả lời dứt điểm câu hỏi này.\n\n"
                "2. Căn cứ pháp luật\n"
                "Không có căn cứ pháp luật cụ thể trong dữ liệu truy xuất.\n\n"
                "3. Phân tích áp dụng\n"
                "Cần bổ sung thêm tình tiết thực tế hoặc diễn đạt truy vấn cụ thể hơn để xác định đúng quy định áp dụng.\n\n"
                "4. Việc SME nên làm\n"
                "- Kiểm tra lại câu hỏi với thông tin chi tiết hơn.\n"
                "- Liên hệ cơ quan chức năng hoặc chuyên gia pháp lý để được tư vấn cụ thể."
            )

        lead = contexts[0]
        lead_meta = dict(lead.get("metadata") or {})
        snippets: List[str] = []
        doc_titles: List[str] = []
        for context in contexts[:3]:
            meta = dict(context.get("metadata") or {})
            text = re.sub(r"\s+", " ", str(context.get("content") or "")).strip()
            if text:
                snippets.append(text[:200].rstrip(" ,;:"))
            title = str(meta.get("doc_title") or "").strip()
            if title and title not in doc_titles:
                doc_titles.append(title)
        
        analysis = " ".join(snippets).strip()
        if not analysis:
            analysis = "Ngữ liệu truy xuất cho thấy có quy định liên quan nhưng chưa đủ dài để trích xuất thêm chi tiết an toàn."

        citations_text = ""
        if citations:
            cite_parts = []
            for c in citations[:3]:
                title = str(c.get("doc_title") or "").strip()
                article = str(c.get("article") or "").strip()
                if title and article:
                    cite_parts.append(f"{title}, {article}")
                elif title:
                    cite_parts.append(title)
            if cite_parts:
                citations_text = "; ".join(cite_parts)

        lead_title = str(lead_meta.get("doc_title") or "").strip()
        doc_list = "; ".join(doc_titles[:2]) if doc_titles else lead_title

        return (
            f"1. Kết luận ngắn\n"
            f"Theo nội dung truy xuất từ {doc_list}, có quy định liên quan đến vấn đề này.\n\n"
            f"2. Căn cứ pháp luật\n"
            f"{citations_text if citations_text else 'Có quy định liên quan trong các văn bản pháp luật đã truy xuất.'}\n\n"
            f"3. Phân tích áp dụng\n"
            f"{analysis}\n\n"
            f"4. Việc SME nên làm\n"
            f"- Đối chiếu nội dung trên với tình tiết thực tế của doanh nghiệp.\n"
            f"- Kiểm tra thời điểm áp dụng và hiệu lực của văn bản pháp luật.\n"
            f"- Chuẩn bị hồ sơ cụ thể trước khi ra quyết định.\n"
            f"- Nếu cần, liên hệ chuyên gia pháp lý để được tư vấn chi tiết."
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
