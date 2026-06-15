import json
import re
from collections import Counter
from pathlib import Path

def analyze_eval_dataset(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    domains = []
    law_refs = []
    question_lengths = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                question_lengths.append(len(data.get("question", "")))
                
                # Extract domains if available
                if "domain" in data:
                    domains.extend(data["domain"])
                elif "domains" in data:
                    domains.extend(data["domains"])
                    
                # Extract expected law refs
                refs = data.get("expected_law_refs", [])
                if isinstance(refs, list):
                    law_refs.extend(refs)
                elif isinstance(refs, str):
                    law_refs.append(refs)
            except json.JSONDecodeError:
                continue

    print("="*60)
    print(f"PHÂN TÍCH DATASET: {path.name}")
    print("="*60)
    print(f"Tổng số câu hỏi: {len(question_lengths)}")
    print(f"Độ dài câu hỏi trung bình: {sum(question_lengths)/len(question_lengths):.0f} ký tự")
    
    print("\n--- TOP 10 DOMAIN PHỔ BIẾN NHẤT ---")
    domain_counts = Counter(domains)
    for domain, count in domain_counts.most_common(10):
        print(f"  {domain}: {count} câu ({count/len(question_lengths)*100:.1f}%)")

    print("\n--- TOP 15 VĂN BẢN PHÁP LUẬT ĐƯỢC KỲ VỌNG (Expected Law Refs) ---")
    # Clean up refs for better grouping (e.g., extract "Luật Doanh nghiệp", "Nghị định 123")
    cleaned_refs = []
    for ref in law_refs:
        # Try to extract the main law name (e.g., "Luật Doanh nghiệp 2020", "Nghị định 123/2020/NĐ-CP")
        match = re.search(r'(Luật|Nghị định|Thông tư|Bộ luật|Pháp lệnh|Quyết định)[^\n,;]+', ref, re.IGNORECASE)
        if match:
            cleaned_refs.append(match.group(0).strip())
        else:
            cleaned_refs.append(ref.strip())
            
    ref_counts = Counter(cleaned_refs)
    for ref, count in ref_counts.most_common(15):
        print(f"  [{count} lần] {ref}")

    print("\n--- GỢI Ý CHIẾN LƯỢC EXPAND DATA ---")
    print("Dựa trên top văn bản pháp luật trên, ưu tiên thu thập/kiểm tra sự hiện diện của:")
    top_5_refs = [ref for ref, _ in ref_counts.most_common(5)]
    for ref in top_5_refs:
        print(f"  -> {ref}")

if __name__ == "__main__":
    analyze_eval_dataset("data/evaluation/r2ai_stage1_questions.jsonl")
