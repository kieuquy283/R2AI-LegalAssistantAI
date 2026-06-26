"""Batch update Qdrant legal_parquet_v2 payload with domain field."""
import json
import os
import time
from collections import Counter, defaultdict
from qdrant_client import QdrantClient

MAPPING_PATH = "data/processed/doc_domain_map.json"
COLLECTION = "legal_parquet_v2"
BATCH_SIZE = 500
DEFAULT_DOMAIN = "business_law"

def main():
    print(f"[DomainUpdate] Loading domain mapping from {MAPPING_PATH}...")
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        doc_domain_map = json.load(f)
    print(f"[DomainUpdate] Loaded {len(doc_domain_map)} mappings")

    # Fallback: rule-based domain detection from doc_title
    # These are from domain_taxonomy.json keywords
    TITLE_KEYWORDS = {
        "business_law": ["doanh nghiệp", "công ty", "thành lập", "đăng ký kinh doanh", "vốn điều lệ",
                         "cổ phần", "tnhh", "thương mại"],
        "tax_law": ["thuế", "hóa đơn", "gtgt", "tndn", "tncn", "kê khai thuế", "môn bài"],
        "labor_law": ["lao động", "người lao động", "hợp đồng lao động", "tiền lương", "bhxh",
                      "bảo hiểm xã hội", "nghỉ việc", "thử việc"],
        "investment_law": ["đầu tư", "fdi", "nhà đầu tư nước ngoài", "irc"],
        "land_law": ["đất đai", "thuê đất", "sử dụng đất", "giấy chứng nhận quyền sử dụng đất",
                     "xây dựng", "nhà ở", "bất động sản"],
        "administrative_penalty": ["xử phạt", "vi phạm hành chính", "mức phạt", "chế tài"],
    }

    def rule_based_domain(title: str) -> str:
        if not title:
            return DEFAULT_DOMAIN
        title_lower = title.lower()
        scores = {}
        for domain, kws in TITLE_KEYWORDS.items():
            scores[domain] = sum(1 for kw in kws if kw in title_lower)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else DEFAULT_DOMAIN

    print(f"[DomainUpdate] Connecting to Qdrant...")
    client = QdrantClient("localhost", port=6333)

    col_info = client.get_collection(COLLECTION)
    total_points = col_info.points_count
    print(f"[DomainUpdate] Collection {COLLECTION}: {total_points} points")

    updated_count = 0
    mapped_count = 0
    rule_count = 0
    default_count = 0
    domain_dist = Counter()
    offset = None
    t0 = time.perf_counter()

    while True:
        batch, offset = client.scroll(
            COLLECTION,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break

        points_to_update = []
        for point in batch:
            doc_num = (point.payload.get("doc_number") or "").strip().replace(" ", "")
            title = point.payload.get("doc_title") or ""

            # Try exact match from HF dataset
            domain = doc_domain_map.get(doc_num)
            if domain:
                src = "hf"
                mapped_count += 1
            else:
                # Fallback: rule-based from title
                domain = rule_based_domain(title)
                src = "rule"
                rule_count += 1

            if domain == DEFAULT_DOMAIN:
                default_count += 1

            domain_dist[domain] += 1
            points_to_update.append((point.id, {"domain": domain}))

        if points_to_update:
            by_domain = defaultdict(list)
            for pid, payload in points_to_update:
                by_domain[payload["domain"]].append(pid)
            for domain, pids in by_domain.items():
                client.overwrite_payload(
                    COLLECTION,
                    payload={"domain": domain},
                    points=pids,
                )
            updated_count += len(points_to_update)

        elapsed = time.perf_counter() - t0
        pct = updated_count / total_points * 100 if total_points else 0
        if updated_count % 2000 == 0 or updated_count == total_points:
            print(f"[DomainUpdate] Updated {updated_count}/{total_points} ({pct:.1f}%) in {elapsed:.1f}s")

        if offset is None:
            break

    t_total = time.perf_counter() - t0
    print(f"\n[Done] Updated {updated_count} points in {t_total:.1f}s")
    print(f"  HF-mapped: {mapped_count} ({mapped_count/updated_count*100:.1f}%)")
    print(f"  Rule-based: {rule_count} ({rule_count/updated_count*100:.1f}%)")
    print(f"  Default: {default_count} ({default_count/updated_count*100:.1f}%)")
    print(f"  Domain distribution:")
    for dom, cnt in domain_dist.most_common():
        print(f"    {dom}: {cnt} ({cnt/updated_count*100:.1f}%)")

    # Save stats
    stats = {
        "total_updated": updated_count,
        "hf_mapped": mapped_count,
        "rule_based": rule_count,
        "default": default_count,
        "time_seconds": round(t_total, 2),
        "domain_distribution": dict(domain_dist.most_common()),
    }
    with open("data/processed/domain_update_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[Stats] Saved to data/processed/domain_update_stats.json")

    # Verify: sample 10 random points
    print("\n[Verify] Sample points with domain:")
    offset = None
    seen = 0
    while seen < 10:
        batch, offset = client.scroll(COLLECTION, limit=100, offset=offset, with_payload=True, with_vectors=False)
        for p in batch:
            if seen >= 10:
                break
            dn = p.payload.get("doc_number", "")
            dom = p.payload.get("domain", "N/A")
            title = (p.payload.get("doc_title") or "")[:60]
            print(f"  [{dn}] domain={dom} -> {title}")
            seen += 1
        if offset is None:
            break

if __name__ == "__main__":
    main()
