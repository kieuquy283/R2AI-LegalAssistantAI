from __future__ import annotations

from typing import Iterable

from legal_rag.aggregation.article import SelectedArticle


class LegalAnswerGenerator:
    def build_evidence_pack(self, articles: Iterable[SelectedArticle]) -> str:
        lines: list[str] = []
        for article in articles:
            excerpt = (article.evidence[0] if article.evidence else "").replace("\n", " ").strip()
            lines.append(f"- {article.article_number} - {article.doc_title}: {excerpt}")
        return "\n".join(lines)

    def generate(self, question: str, articles: list[SelectedArticle]) -> str:
        if not articles:
            return (
                "Kết luận ngắn: Chưa đủ căn cứ pháp lý trong tập tài liệu đã truy xuất để trả lời chắc chắn.\n\n"
                "Căn cứ pháp lý: Hiện chưa truy xuất được Điều cụ thể phù hợp với câu hỏi.\n\n"
                "Giải thích dễ hiểu: Cần bổ sung văn bản hoặc từ khóa truy vấn liên quan hơn để tránh suy đoán ngoài căn cứ.\n\n"
                "Gợi ý áp dụng thực tế: Doanh nghiệp nên rà lại tên văn bản, số điều hoặc bối cảnh nghiệp vụ trước khi kết luận.\n\n"
                "Lưu ý: Đây là tư vấn sơ bộ dựa trên các căn cứ được cung cấp."
            )

        lead = articles[0]
        legal_basis = "; ".join(f"{article.article_number} {article.doc_title}" for article in articles[:3])
        explanation = lead.evidence[0].split("\n", 1)[-1].strip() if lead.evidence else ""
        if not explanation:
            explanation = lead.article_title or "Căn cứ chính nằm trong điều luật đã được truy xuất."

        return (
            f"Kết luận ngắn: Theo {lead.article_number} {lead.doc_title}, câu hỏi được điều chỉnh trực tiếp bởi căn cứ pháp lý đã truy xuất.\n\n"
            f"Căn cứ pháp lý: Căn cứ {legal_basis}.\n\n"
            f"Giải thích dễ hiểu: {explanation}\n\n"
            "Gợi ý áp dụng thực tế: Doanh nghiệp nên đối chiếu hồ sơ, quy trình nội bộ và văn bản liên quan để áp dụng đúng điều khoản nêu trên.\n\n"
            "Lưu ý: Đây là tư vấn sơ bộ dựa trên các căn cứ được cung cấp; không bổ sung điều luật ngoài evidence."
        )
