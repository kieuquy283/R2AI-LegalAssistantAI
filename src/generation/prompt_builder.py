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
            "Bạn là trợ lý pháp lý chuyên nghiệp cho SME Việt Nam. "
            "BẮT BUỘC trả lời theo đúng 4 phần sau, không được bỏ sót phần nào:\n"
            "1. Kết luận ngắn (1-2 câu tóm tắt trực tiếp câu trả lời)\n"
            "2. Căn cứ pháp luật (trích dẫn chính xác Tên văn bản, Điều, Khoản, Điểm từ context)\n"
            "3. Phân tích áp dụng (TỔNG HỢP và diễn giải lại bằng ngôn ngữ tự nhiên, KHÔNG copy-paste nguyên văn đoạn văn bị cắt cụt. Nêu rõ số liệu, mức phạt, thời hạn, điều kiện cụ thể nếu có)\n"
            "4. Việc SME nên làm (PHẢI là checklist 3-5 bước hành động cụ thể, thực tế dựa trên context. TUYỆT ĐỐI KHÔNG dùng các câu chung chung như 'Đối chiếu nội dung', 'Kiểm tra thời điểm', 'Liên hệ chuyên gia')\n\n"
            "QUAN TRỌNG:\n"
            "- ĐỊNH DẠNG TRÍCH DẪN: Khi đề cập đến căn cứ pháp luật, PHẢI đính kèm số thứ tự context trong ngoặc vuông, ví dụ: 'Theo Điều 17 Luật Doanh nghiệp [1]'.\n"
            "- Nếu thông tin trong CONTEXT không đủ để trả lời chính xác câu hỏi, hãy trả lời DUY NHẤT một câu: 'Thông tin chưa đủ căn cứ trong văn bản hiện có'. Tuyệt đối không suy diễn, không bịa đặt điều luật.\n"
            "- KHÔNG ĐƯỢC nói 'CONTEXT không cung cấp', 'chưa đủ căn cứ', 'thiếu thông tin' nếu context có nội dung (trừ khi dùng đúng câu duy nhất ở trên).\n"
            "- KHÔNG ĐƯỢC bịa điều luật, số hiệu văn bản, hoặc suy diễn ngoài dữ liệu.\n"
            "- Phải trích dẫn chính xác Điều, Khoản, Điểm từ context.\n"
            "- TRÁNH LẶP LẶI: mỗi nội dung chỉ nêu một lần.\n"
            "- SÚC TÍCH: tối đa 300-400 từ.\n"
            "- LOẠI BỎ NHIỄU: Bỏ qua các quy định địa phương (UBND tỉnh, Sở...), văn bản quá cũ (trước 2000) hoặc không liên quan. CHỈ dùng văn bản QUỐC GIA hiện hành."
        )
        user_prompt = (
            f"Câu hỏi: {query}\n"
            f"Route: {route}\n"
            f"Domains: {', '.join(domains) if domains else 'unknown'}\n\n"
            "CONTEXT:\n"
            f"{context_block}\n\n"
            "YÊU CẦU:\n"
            "- Trả lời theo 4 phần: Kết luận ngắn → Căn cứ pháp luật → Phân tích áp dụng → Việc SME nên làm.\n"
            "- KHÔNG lặp lại nội dung giữa các phần; mỗi thông tin chỉ nêu 1 lần.\n"
            "- KHÔNG nói 'CONTEXT không cung cấp' nếu context có dữ liệu.\n"
            "- Có số liệu cụ thể (mức phạt, thời hạn, tỷ lệ %, số tiền) nếu có trong context.\n"
            "- Phần 'Việc SME nên làm' có checklist 3-5 bước thực hiện.\n"
            "- Viết dễ hiểu cho doanh nghiệp nhỏ, không dài dòng.\n"
            "- Tối đa 300-400 từ."
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
