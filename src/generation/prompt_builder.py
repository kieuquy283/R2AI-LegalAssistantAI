from __future__ import annotations

import argparse
import json
from typing import Dict, List


class PromptBuilder:
    def build(
        self,
        *,
        query: str,
        contexts: List[Dict[str, object]],
        route: str,
        domains: List[str],
    ) -> Dict[str, str]:
        context_lines: List[str] = []
        for index, context in enumerate(contexts, start=1):
            metadata = dict(context.get("metadata") or {})
            citation = metadata.get("citation") or f"{metadata.get('doc_title') or ''} {metadata.get('article') or ''}".strip()
            context_lines.append(
                "\n".join(
                    [
                        f"[Context {index}]",
                        f"Citation: {citation or 'N/A'}",
                        f"Domain: {metadata.get('domain') or 'unknown'}",
                        f"Source: {metadata.get('source_url') or 'N/A'}",
                        f"Content: {str(context.get('content') or '').strip()}",
                    ]
                )
            )
        context_block = "\n\n".join(context_lines) if context_lines else "No context available."
        system_prompt = (
            "Bạn là trợ lý pháp lý cho SME Việt Nam. "
            "Chỉ trả lời dựa trên CONTEXT. "
            "Không bịa điều luật, không bịa số hiệu văn bản, không suy diễn ngoài dữ liệu đã truy xuất. "
            "Nếu thiếu căn cứ, phải nói chưa đủ căn cứ. "
            "Phải nêu căn cứ pháp luật nếu context có và phân biệt căn cứ chính với căn cứ liên quan khi có nhiều domain."
        )
        user_prompt = (
            f"Câu hỏi: {query}\n"
            f"Route: {route}\n"
            f"Domains: {', '.join(domains) if domains else 'unknown'}\n\n"
            "Yêu cầu trả lời theo đúng format sau:\n"
            "1. Kết luận ngắn\n"
            "2. Căn cứ pháp luật\n"
            "3. Phân tích áp dụng vào tình huống\n"
            "4. Việc SME nên làm\n"
            "5. Lưu ý/rủi ro\n\n"
            "CONTEXT:\n"
            f"{context_block}\n\n"
            "Lưu ý:\n"
            "- Chỉ dùng thông tin xuất hiện trong CONTEXT.\n"
            "- Nếu context thiếu, nêu rõ chưa đủ căn cứ.\n"
            "- Cố gắng viết dễ hiểu cho SME."
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context_block": context_block,
        }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build legal QA prompt from contexts.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--route", default="SIMPLE_VECTOR")
    parser.add_argument("--domains", action="append", default=None)
    parser.add_argument("--contexts-json", required=True)
    args = parser.parse_args()
    builder = PromptBuilder()
    result = builder.build(
        query=args.query,
        contexts=json.loads(args.contexts_json),
        route=args.route,
        domains=args.domains or [],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
