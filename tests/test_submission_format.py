from legal_rag.submission.validator import validate_submission_payload


def test_submission_validator_accepts_valid_payload():
    payload = [
        {
            "id": 1,
            "question": "Quy định nào áp dụng?",
            "answer": "Kết luận ngắn: ... Căn cứ pháp lý: Căn cứ Điều 4 Luật Ban hành văn bản quy phạm pháp luật.",
            "relevant_docs": ["64/2025/QH15|Luật Ban hành văn bản quy phạm pháp luật"],
            "relevant_articles": ["64/2025/QH15|Luật Ban hành văn bản quy phạm pháp luật|Điều 4"],
        }
    ]

    report = validate_submission_payload(payload)

    assert report.ok is True
    assert report.errors == []


def test_submission_validator_detects_invalid_format():
    payload = [
        {
            "id": 1,
            "question": "Quy định nào áp dụng?",
            "answer": "Không có điều luật nào được nêu.",
            "relevant_docs": ["sai_format"],
            "relevant_articles": ["64/2025/QH15|Luật Ban hành văn bản quy phạm pháp luật|Điều 4"],
        }
    ]

    report = validate_submission_payload(payload)

    assert report.ok is False
    assert any("invalid relevant_docs format" in error for error in report.errors)
