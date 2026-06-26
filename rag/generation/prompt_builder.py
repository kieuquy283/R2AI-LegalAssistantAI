from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document

from rag.config.retrieval import HISTORY_TURNS


INSUFFICIENT_SENTINEL = "__INSUFFICIENT_CONTEXT__"


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "Không tìm thấy tài liệu liên quan."

    formatted_parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        content = doc.page_content.strip()

        block = (
            f"[Document {i} | Source: {source} | Page: {page} | Chunk: {chunk_id}]\n"
            f"{content}"
        )
        formatted_parts.append(block)

    return "\n\n".join(formatted_parts)


def format_recent_history(
    history: List[Dict[str, Any]],
    max_turns: int = HISTORY_TURNS,
) -> str:
    if not history:
        return "No previous conversation."

    selected = history[-max_turns * 2 :]
    lines = []

    for msg in selected:
        role = str(msg.get("role", "user")).capitalize()
        content = str(msg.get("content", "")).strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def build_answer_prompt(
    current_question: str,
    rewritten_query: str,
    docs: List[Document],
    history: List[Dict[str, Any]],
    max_turns: int = HISTORY_TURNS,
) -> str:
    recent_history = format_recent_history(history, max_turns=max_turns)
    context = format_docs(docs)

    prompt = f"""
Bạn là trợ lý AI đóng vai trò tư vấn pháp lý cho doanh nghiệp trả lời câu hỏi dựa trên tài liệu được truy xuất.

Nguyên tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong phần Context.
2. Nếu Context không đủ thông tin để trả lời chính xác và chắc chắn câu hỏi của người dùng, bạn PHẢI trả lời theo mẫu định dạng chuẩn sau đây:

### PHẢN HỒI VỀ YÊU CẦU TRA CỨU PHÁP LÝ

**Vấn đề tra cứu:** *“[Câu hỏi của người dùng]”*

---

#### 1. Kết quả rà soát dữ liệu
Sau khi tiến hành kiểm tra hệ thống tài liệu được cung cấp (bao gồm: [Danh sách các văn bản pháp luật tìm thấy trong Context, ví dụ: Nghị định 121/2026/NĐ-CP,...]), chúng tôi ghi nhận kết quả như sau:
* Các văn bản nêu trên **không đề cập** đến [Nêu vấn đề còn thiếu trong tài liệu].
* **Không có quy định** về [Nêu quy định cụ thể còn thiếu liên quan đến câu hỏi].

#### 2. Kết luận từ nguồn tài liệu (Context)
Do giới hạn thông tin của nguồn dữ liệu được cung cấp, chúng tôi chưa thể đưa ra câu trả lời chính xác cho câu hỏi của bạn dựa trên các tài liệu này.

#### 3. Hướng dẫn tham khảo thêm
Để có thông tin giải đáp đầy đủ và chính xác nhất cho trường hợp này, bạn nên tham khảo thêm các văn bản pháp luật chuyên ngành có liên quan trực tiếp, ví dụ: [Nêu các văn bản luật chuyên ngành ngoài Context phù hợp].

3. Không bịa, không suy đoán vượt quá dữ liệu.
4. Trả lời rõ ràng, đúng trọng tâm, dễ hiểu.

Lịch sử hội thoại gần đây:
{recent_history}

Câu hỏi hiện tại của người dùng:
{current_question}

Truy vấn độc lập dùng để tìm kiếm:
{rewritten_query}

Context:
{context}

Hãy trả lời bằng tiếng Việt.
""".strip()

    return prompt


def build_fallback_prompt(
    question: str,
    history: List[Dict[str, Any]],
) -> str:
    history_text = ""
    if history:
        recent = history[-HISTORY_TURNS * 2 :]
        history_text = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in recent
        )

    return f"""
Bạn là trợ lý AI hữu ích.

Kho tri thức hiện tại không có đủ tài liệu liên quan hoặc không đủ ngữ cảnh để trả lời chắc chắn câu hỏi của người dùng.

Hãy trả lời câu hỏi bằng kiến thức nền của bạn một cách hữu ích, rõ ràng, trung thực.
Không được nói rằng bạn có tài liệu nếu thực tế không có.
Nếu có điểm chưa chắc chắn, hãy nói rõ.

Lịch sử hội thoại gần đây:
{history_text if history_text else "Không có."}

Câu hỏi hiện tại:
{question}
""".strip()


def build_grounded_prompt_with_guardrail(
    current_question: str,
    rewritten_query: str,
    docs: List[Document],
    history: List[Dict[str, Any]],
) -> str:
    base_prompt = build_answer_prompt(
        current_question=current_question,
        rewritten_query=rewritten_query,
        docs=docs,
        history=history,
        max_turns=HISTORY_TURNS,
    )

    extra_rule = f"""

QUY TẮC BẮT BUỘC:
- Chỉ dùng thông tin từ phần tài liệu được cung cấp.
- Nếu tài liệu không đủ để trả lời chắc chắn, hãy chỉ in đúng duy nhất chuỗi sau:
{INSUFFICIENT_SENTINEL}
- Không giải thích thêm khi in chuỗi đó.
"""

    return base_prompt + "\n\n" + extra_rule
