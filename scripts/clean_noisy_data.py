import json
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Danh sách các mẫu (regex) xác định dữ liệu nhiễu dựa trên lỗi eval trước đó
NOISE_PATTERNS = [
    r"UBND TỈNH",
    r"UBND THÀNH PHỐ",
    r"SỞ TƯ PHÁP",
    r"SỞ KHOA HỌC VÀ CÔNG NGHỆ",
    r"LÂM ĐỒNG",
    r"SẮC LỆNH",
    r"HỘI ĐỒNG BỘ TRƯỞNG",
    r"NĂM 19[5-9][0-9]",  # Các năm từ 1950 đến 1999
    r"QUYẾT ĐỊNH CỦA CHỦ TỊCH UBND",
]

def is_noisy(text: str) -> bool:
    """Kiểm tra xem văn bản có chứa mẫu nhiễu hay không."""
    text_upper = text.upper()
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text_upper):
            return True
    return False

def clean_jsonl_file(input_path: Path, output_path: Path) -> None:
    """Đọc file JSONL, lọc bỏ các dòng nhiễu và ghi ra file mới."""
    if not input_path.exists():
        logger.warning("File không tồn tại: %s", input_path)
        return

    total_lines = 0
    kept_lines = 0
    removed_lines = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            total_lines += 1
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Dòng không hợp lệ JSON, bỏ qua: %d", total_lines)
                continue

            # Kiểm tra nhiễu trong title, doc_title, hoặc content
            text_to_check = f"{record.get('title', '')} {record.get('doc_title', '')} {record.get('content', '')}"
            
            if is_noisy(text_to_check):
                removed_lines += 1
            else:
                outfile.write(line)
                kept_lines += 1

    logger.info("=== Kết quả lọc: %s ===", input_path.name)
    logger.info("Tổng số dòng: %d", total_lines)
    logger.info("Đã giữ lại (sạch): %d", kept_lines)
    logger.info("Đã loại bỏ (nhiễu): %d", removed_lines)
    logger.info("File đầu ra: %s", output_path)

def main():
    processed_dir = Path("data/processed")
    
    files_to_clean = [
        ("merged_documents.jsonl", "merged_documents_clean.jsonl"),
        ("merged_legal_nodes.jsonl", "merged_legal_nodes_clean.jsonl"),
    ]
    
    for input_name, output_name in files_to_clean:
        input_path = processed_dir / input_name
        output_path = processed_dir / output_name
        clean_jsonl_file(input_path, output_path)

    logger.info("Hoàn tất quá trình làm sạch dữ liệu. Bạn có thể dùng các file *_clean.jsonl để upsert lại vào Qdrant.")

if __name__ == "__main__":
    main()
