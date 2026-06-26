"""Build doc_number → domain mapping from HF dataset legacy/metadata + domain_taxonomy."""
import json
import os
from collections import Counter
from datasets import load_dataset

MAPPING_OUT = "data/processed/doc_domain_map.json"
STATS_OUT = "data/processed/doc_domain_stats.json"

# legal_sectors → taxonomy domain mapping
SECTOR_TO_DOMAIN = {
    "Finance": "tax_law",
    "Tax": "tax_law",
    "Taxes": "tax_law",
    "Accounting": "tax_law",
    "Audit": "accounting_law",
    "Labor": "labor_law",
    "Employment": "labor_law",
    "Wages": "labor_law",
    "Social Insurance": "social_insurance",
    "Health Insurance": "social_insurance",
    "Unemployment Insurance": "social_insurance",
    "Enterprise": "business_law",
    "Business": "business_law",
    "Corporate": "business_law",
    "Company": "business_law",
    "Investment": "investment_law",
    "Foreign Investment": "investment_law",
    "Trade": "investment_law",
    "Commerce": "civil_commercial_law",
    "Contracts": "civil_commercial_law",
    "Intellectual Property": "ip_law",
    "Copyright": "ip_law",
    "Trademark": "ip_law",
    "Land": "land_law",
    "Construction": "land_law",
    "Urban Planning": "land_law",
    "Environment": "land_law",
    "Natural Resources": "land_law",
    "Administrative": "administrative_penalty",
    "Penalty": "administrative_penalty",
    "Violations": "administrative_penalty",
    "Education": "business_law",
    "Training": "business_law",
    "Health": "business_law",
    "Medical": "business_law",
    "Insurance": "social_insurance",
    "Information Technology": "business_law",
    "Science": "business_law",
    "Technology": "business_law",
    "Transport": "business_law",
    "Agriculture": "business_law",
    "Fisheries": "business_law",
    "Culture": "business_law",
    "Tourism": "business_law",
    "Banking": "tax_law",
    "Credit": "tax_law",
    "Securities": "tax_law",
}

DEFAULT_DOMAIN = "business_law"

def normalize_sector(s: str) -> str:
    """Normalize sector string for matching."""
    return s.strip().lower().replace("–", "-").replace("—", "-")

def sector_to_domain(sectors: list[str]) -> str:
    """Map first matching sector to taxonomy domain."""
    if not sectors:
        return DEFAULT_DOMAIN
    for s in sectors:
        s_norm = normalize_sector(s)
        for key, domain in SECTOR_TO_DOMAIN.items():
            if normalize_sector(key) == s_norm or key.lower() in s_norm or s_norm in key.lower():
                return domain
    return DEFAULT_DOMAIN

def main():
    print("[DomainMap] Loading metadata from HF dataset (streaming)...")
    meta = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        "legacy",
        split="metadata",
        streaming=True,
    )
    print(f"[DomainMap] Streaming started")

    domain_map = {}
    sector_counter = Counter()
    domain_counter = Counter()
    no_sector = 0
    count = 0

    for row in meta:
        count += 1
        doc_num = row.get("document_number")
        sectors = row.get("legal_sectors")
        if not doc_num:
            continue
        # legal_sectors can be None, string, or list
        if sectors is None:
            sector_list = []
            no_sector += 1
        elif isinstance(sectors, str):
            sector_list = [s.strip() for s in sectors.split(",") if s.strip()]
        elif isinstance(sectors, (list, tuple)):
            sector_list = [s.strip() for s in sectors if s and str(s).strip()]
        else:
            sector_list = []

        # Normalize doc_num
        doc_num = doc_num.strip().replace(" ", "")
        if not doc_num:
            continue

        for s in sector_list:
            sector_counter[normalize_sector(s)] += 1

        domain = sector_to_domain(sector_list)
        domain_counter[domain] += 1
        domain_map[doc_num] = domain

    print(f"[DomainMap] Mapped {len(domain_map)} unique doc_numbers")
    print(f"[DomainMap] Documents without sectors: {no_sector}")
    print(f"[DomainMap] Domain distribution: {dict(domain_counter.most_common())}")
    print(f"[DomainMap] Top 10 sectors: {dict(sector_counter.most_common(10))}")

    # Save mapping
    os.makedirs(os.path.dirname(MAPPING_OUT), exist_ok=True)
    with open(MAPPING_OUT, "w", encoding="utf-8") as f:
        json.dump(domain_map, f, ensure_ascii=False, indent=2)
    print(f"[DomainMap] Saved mapping to {MAPPING_OUT}")

    stats = {
        "total": len(domain_map),
        "no_sector_in_source": no_sector,
        "domain_distribution": dict(domain_counter.most_common()),
        "top_sectors": dict(sector_counter.most_common(20)),
    }
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[DomainMap] Saved stats to {STATS_OUT}")

if __name__ == "__main__":
    main()
