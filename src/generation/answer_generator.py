from __future__ import annotations

import argparse
import json
from typing import Dict, List

from src.retrieval.retrieval_pipeline import RetrievalPipeline


class AnswerGenerator:
    def __init__(self, *, temperature: float = 0.1) -> None:
        self.temperature = float(temperature)

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

    def generate(self, *, query: str, retrieval_result: Dict[str, object]) -> Dict[str, object]:
        contexts = list(retrieval_result.get("final_contexts") or [])
        citations = self._collect_citations(contexts)
        used_context_ids = [str(context["chunk_id"]) for context in contexts]

        if not contexts:
            answer = (
                "1. Kết luận ngắn: Chưa đủ căn cứ để kết luận chắc chắn.\n"
                "2. Căn cứ pháp luật: Hiện chưa có điều/khoản phù hợp trong context được truy xuất.\n"
                "3. Phân tích áp dụng vào tình huống: Nếu thiếu context thì không nên suy diễn thêm căn cứ pháp luật.\n"
                "4. Hướng xử lý thực tế cho SME: Cần bổ sung thông tin thực tế hoặc truy vấn cụ thể hơn.\n"
                "5. Lưu ý nếu thiếu dữ kiện: Câu trả lời này chỉ phản ánh việc chưa đủ căn cứ trong dữ liệu đã truy xuất."
            )
            return {"answer": answer, "citations": [], "used_context_ids": used_context_ids}

        lead = contexts[0]
        lead_meta = dict(lead.get("metadata") or {})
        legal_basis = "; ".join(
            citation.get("citation") or f"{citation.get('doc_title')} {citation.get('article')}"
            for citation in citations[:3]
            if citation.get("doc_title")
        )
        snippets = []
        for context in contexts[:2]:
            text = " ".join(str(context.get("content") or "").split())
            snippets.append(text[:260])
        analysis = " ".join(snippets).strip()
        if not analysis:
            analysis = "Context hiện có chưa đủ dài để trích đoạn rõ hơn nhưng vẫn cho thấy căn cứ chính nằm ở điều luật đã truy xuất."

        answer = (
            f"1. Kết luận ngắn: Câu hỏi này trước hết được điều chỉnh bởi {lead_meta.get('citation') or lead_meta.get('article') or 'căn cứ pháp luật đã truy xuất'}.\n"
            f"2. Căn cứ pháp luật: {legal_basis or 'Chưa đủ căn cứ pháp luật cụ thể trong context hiện có.'}\n"
            f"3. Phân tích áp dụng vào tình huống: {analysis}\n"
            "4. Hướng xử lý thực tế cho SME: Doanh nghiệp nên đối chiếu hồ sơ, thời điểm áp dụng và nghĩa vụ cụ thể với điều/khoản đã nêu trước khi ra quyết định.\n"
            "5. Lưu ý nếu thiếu dữ kiện: Câu trả lời chỉ dựa trên context đã truy xuất, không bổ sung thêm căn cứ pháp luật ngoài dữ liệu hiện có."
        )
        return {"answer": answer, "citations": citations, "used_context_ids": used_context_ids}


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
