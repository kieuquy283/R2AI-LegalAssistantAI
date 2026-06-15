import json
import re
from collections import Counter
from pathlib import Path

def analyze_questions_by_domain(file_path: str, taxonomy_path: str):
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    
    domain_counts = Counter()
    question_domains = {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            q_id = data.get("id")
            question = data.get("question", "").lower()
            
            matched_domains = []
            for domain, info in taxonomy.items():
                keywords = info.get("keywords", [])
                # Check if any keyword is in the question
                for kw in keywords:
                    if kw.lower() in question:
                        matched_domains.append(domain)
                        break
            
            if matched_domains:
                for d in matched_domains:
                    domain_counts[d] += 1
                question_domains[q_id] = matched_domains
            else:
                domain_counts["unclassified"] += 1
                question_domains[q_id] = ["unclassified"]

    total_questions = sum(domain_counts.values())
    print("="*70)
    print(f"PHÂN TÍCH DOMAIN CHO {total_questions} CÂU HỎI TRONG: {Path(file_path).name}")
    print("="*70)
    
    print("\n--- PHÂN BỐ DOMAIN (Có thể trùng lặp do đa domain) ---")
    for domain, count in domain_counts.most_common():
        pct = (count / total_questions) * 100
        print(f"  {domain:<25}: {count:>4} câu ({pct:>5.1f}%)")

    print("\n--- CHIẾN LƯỢC EXPAND DATA DỰA TRÊN PHÂN TÍCH ---")
    top_domains = [d for d, _ in domain_counts.most_common(5) if d != "unclassified"]
    print("1. Ưu tiên thu thập / làm giàu dữ liệu cho các domain top đầu:")
    for d in top_domains:
        print(f"   - {d}: {taxonomy[d]['description']}")
        print(f"     Keywords: {', '.join(taxonomy[d]['keywords'][:5])}...")
    
    print("\n2. Hành động cụ thể:")
    print("   - Chạy lại `filter_hf_legal_dataset.py` với trọng số cao cho các domain trên.")
    print("   - Bổ sung crawl từ LuatVietnam cho các văn bản: Luật Doanh nghiệp, Luật Đất đai, Luật Lao động, các Nghị định xử phạt vi phạm hành chính mới nhất.")
    print("   - Đảm bảo các document này có metadata `domain` chính xác để Hybrid Reranker có thể boost điểm.")

if __name__ == "__main__":
    analyze_questions_by_domain(
        "data/evaluation/r2ai_stage1_questions.jsonl",
        "data/sources/domain_taxonomy.json"
    )
