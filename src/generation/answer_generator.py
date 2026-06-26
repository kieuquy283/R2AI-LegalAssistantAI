from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List

from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.retrieval_pipeline import RetrievalPipeline


_ARTICLE_RE = re.compile(r"(?:Điều|điều)\s+\d+[a-zA-Z]?", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"(?:Khoản|khoản)\s+\d+[a-zA-Z]?", re.IGNORECASE)
_POINT_RE = re.compile(r"(?:Điểm|điểm)\s+[a-zA-Z]", re.IGNORECASE)

_USE_STRUCTURED_OUTPUT = os.getenv("R2AI_USE_STRUCTURED_OUTPUT", "true").strip().lower() in {"1", "true", "yes"}


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

    def _extract_key_refs(self, text: str) -> List[str]:
        """Extract article/clause/point references from text."""
        refs = []
        for match in _ARTICLE_RE.finditer(text):
            refs.append(match.group(0).strip())
        for match in _CLAUSE_RE.finditer(text):
            refs.append(match.group(0).strip())
        for match in _POINT_RE.finditer(text):
            refs.append(match.group(0).strip())
        return refs

    def _summarize_context(self, context: Dict[str, object], index: int) -> str:
        """Extract a concise summary from a single context chunk."""
        metadata = dict(context.get("metadata") or {})
        content = re.sub(r"\s+", " ", str(context.get("content") or "")).strip()
        article = str(metadata.get("article") or "").strip()
        doc_title = str(metadata.get("doc_title") or "").strip()

        # Find the most relevant sentence in the content
        sentences = re.split(r"(?<=[.!?])\s+", content)
        key_sentence = ""
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30 and any(kw in sent.lower() for kw in ("phải", "được", "không được", "quy định", "mức phạt", "thời hạn", "điều kiện")):
                key_sentence = sent[:300]
                break
        if not key_sentence and sentences:
            key_sentence = sentences[0][:300]

        ref = f"{article} {doc_title}".strip() if article else doc_title
        if key_sentence:
            return f"[{index}] {ref}: {key_sentence}"
        return f"[{index}] {ref}"

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

        # Build structured citation list
        cite_lines: List[str] = []
        seen_cites: set[str] = set()
        for i, c in enumerate(citations[:5], start=1):
            title = str(c.get("doc_title") or "").strip()
            article = str(c.get("article") or "").strip()
            citation_str = str(c.get("citation") or "").strip()
            label = article or citation_str or title
            if not label or label in seen_cites:
                continue
            seen_cites.add(label)
            if article and title:
                cite_lines.append(f"[{i}] {article}, {title}")
            elif title:
                cite_lines.append(f"[{i}] {title}")
        citations_block = "\n".join(cite_lines) if cite_lines else "Không truy xuất được citation cụ thể."

        # Summarize top contexts
        summaries: List[str] = []
        for i, ctx in enumerate(contexts[:3], start=1):
            summary = self._summarize_context(ctx, i)
            if summary:
                summaries.append(summary)
        analysis_block = "\n\n".join(summaries) if summaries else "Nội dung truy xuất liên quan nhưng chưa đủ chi tiết để phân tích sâu."

        # Collect all article refs for the conclusion
        all_refs: List[str] = []
        for ctx in contexts[:3]:
            meta = dict(ctx.get("metadata") or {})
            article = str(meta.get("article") or "").strip()
            if article and article not in all_refs:
                all_refs.append(article)
        ref_text = ", ".join(all_refs[:3]) if all_refs else "các quy định pháp luật liên quan"

        # Build 4-section answer
        return (
            f"1. Kết luận ngắn\n"
            f"Dựa trên dữ liệu truy xuất, vấn đề được điều chỉnh bởi {ref_text}. "
            f"Thông tin chi tiết được trình bày ở phần phân tích bên dưới.\n\n"
            f"2. Căn cứ pháp luật\n"
            f"{citations_block}\n\n"
            f"3. Phân tích áp dụng\n"
            f"{analysis_block}\n\n"
            f"4. Việc SME nên làm\n"
            f"- Đối chiếu tình tiết thực tế với các điều khoản nêu trên [1]-[{len(cite_lines) or 1}].\n"
            f"- Xác định thời điểm áp dụng và hiệu lực của văn bản pháp luật.\n"
            f"- Chuẩn bị hồ sơ, giấy tờ liên quan trước khi thực hiện.\n"
            f"- Nếu có vướng mắc, liên hệ chuyên gia pháp lý hoặc cơ quan chức năng để được hướng dẫn cụ thể."
        )

    def _generate_structured(self, system_prompt: str, user_prompt: str) -> Dict[str, str] | None:
        """Generate with JSON mode, parse sections from the JSON response."""
        json_prompt = (
            "Trả về kết quả dưới dạng JSON hợp lệ với schema sau:\n"
            "{\n"
            '  "ket_luan_ngan": "nội dung kết luận ngắn",\n'
            '  "can_cu_phap_luat": "nội dung căn cứ pháp luật",\n'
            '  "phan_tich_ap_dung": "nội dung phân tích áp dụng",\n'
            '  "viec_sme_nen_lam": "nội dung việc SME nên làm"\n'
            "}\n"
            "Chỉ trả về JSON, không thêm text nào khác.\n\n"
            f"{user_prompt}"
        )
        text = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=json_prompt,
            temperature=self.temperature,
        )
        if not text:
            return None
        text = text.strip()
        # Try to parse as JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and all(k in parsed for k in ("ket_luan_ngan",)):
                sections = {
                    "ket_luan_ngan": str(parsed.get("ket_luan_ngan", "")),
                    "can_cu_phap_luat": str(parsed.get("can_cu_phap_luat", "")),
                    "phan_tich_ap_dung": str(parsed.get("phan_tich_ap_dung", "")),
                    "viec_sme_nen_lam": str(parsed.get("viec_sme_nen_lam", "")),
                }
                formatted = (
                    f"1. Kết luận ngắn\n{sections['ket_luan_ngan']}\n\n"
                    f"2. Căn cứ pháp luật\n{sections['can_cu_phap_luat']}\n\n"
                    f"3. Phân tích áp dụng\n{sections['phan_tich_ap_dung']}\n\n"
                    f"4. Việc SME nên làm\n{sections['viec_sme_nen_lam']}"
                )
                return {"answer": formatted, **sections}
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: try to extract JSON from code fence
        m = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return None

    def _parse_structured_sections(self, text: str) -> Dict[str, str] | None:
        """Parse 4 sections from free-text answer into structured dict."""
        sections = {}
        section_patterns = [
            (r"1\.\s*Kết luận ngắn\s*\n(.*?)(?=\n\s*2\.|\Z)", "ket_luan_ngan"),
            (r"2\.\s*Căn cứ pháp luật\s*\n(.*?)(?=\n\s*3\.|\Z)", "can_cu_phap_luat"),
            (r"3\.\s*Phân tích áp dụng\s*\n(.*?)(?=\n\s*4\.|\Z)", "phan_tich_ap_dung"),
            (r"4\.\s*Việc SME nên làm\s*\n(.*)", "viec_sme_nen_lam"),
        ]
        for pattern, key in section_patterns:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                sections[key] = m.group(1).strip()
        if len(sections) == 4:
            return sections
        return None

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
        generation_mode = "template"
        if use_llm:
            if _USE_STRUCTURED_OUTPUT:
                structured = self._generate_structured(
                    system_prompt=prompt["system_prompt"],
                    user_prompt=prompt["user_prompt"],
                )
                if structured:
                    llm_answer = structured["answer"]
                    generation_mode = "llm_structured"
                    print(f"[AnswerGen] Structured JSON mode (contexts={len(contexts)})")
                else:
                    pass
            if llm_answer is None:
                llm_answer = self.llm_client.generate(
                    system_prompt=prompt["system_prompt"],
                    user_prompt=prompt["user_prompt"],
                    temperature=self.temperature,
                )
                if llm_answer:
                    generation_mode = "llm"
        if not llm_answer:
            generation_mode = "template"
            print(f"[AnswerGen] LLM unavailable, using template fallback (contexts={len(contexts)})")
        answer = llm_answer or self._fallback_answer(contexts=contexts, citations=citations)
        return {
            "answer": answer,
            "citations": citations,
            "used_context_ids": used_context_ids,
            "generation_mode": generation_mode,
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
