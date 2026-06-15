import re

# Từ điển ánh xạ các từ viết tắt/thông dụng sang thuật ngữ đầy đủ trong pháp lý/SME
EXPANSION_MAP = {
    r"\bSME\b": "doanh nghiệp nhỏ và vừa",
    r"\bDN\b": "doanh nghiệp",
    r"\bBHXH\b": "bảo hiểm xã hội",
    r"\bBHYT\b": "bảo hiểm y tế",
    r"\bBHTN\b": "bảo hiểm thất nghiệp",
    r"\bHĐLĐ\b": "hợp đồng lao động",
    r"\bNLĐ\b": "người lao động",
    r"\bNSDLĐ\b": "người sử dụng lao động",
    r"\bGTGT\b": "giá trị gia tăng",
    r"\bTNDN\b": "thuế thu nhập doanh nghiệp",
    r"\bTNCN\b": "thuế thu nhập cá nhân",
    r"\bTNHH\b": "trách nhiệm hữu hạn",
    r"\bCTCP\b": "cổ phần",
    r"\bDNTN\b": "doanh nghiệp tư nhân",
}

def expand_query(query: str) -> str:
    """
    Mở rộng câu truy vấn bằng cách thêm thuật ngữ đầy đủ sau các từ viết tắt.
    Ví dụ: "SME bị phạt BHXH" -> "SME doanh nghiệp nhỏ và vừa bị phạt BHXH bảo hiểm xã hội"
    """
    expanded = query
    for abbr, full_text in EXPANSION_MAP.items():
        # Nếu tìm thấy từ viết tắt, thêm thuật ngữ đầy đủ vào ngay sau nó
        if re.search(abbr, query, re.IGNORECASE):
            # \g<0> giữ nguyên từ viết tắt gốc, thêm khoảng trắng và thuật ngữ đầy đủ
            expanded = re.sub(abbr, f"\\g<0> {full_text}", expanded, flags=re.IGNORECASE)
    
    return expanded.strip()
