from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List


TAXONOMY_PATH = Path("data/sources/domain_taxonomy.json")

SIMPLE_VECTOR = "SIMPLE_VECTOR"
PARENT_CONTEXT = "PARENT_CONTEXT"
LEGAL_GRAPH_CONTEXT = "LEGAL_GRAPH_CONTEXT"
CROSS_DOMAIN_CONTEXT = "CROSS_DOMAIN_CONTEXT"
MULTI_DOMAIN_COMPLEX = "MULTI_DOMAIN_COMPLEX"


def _load_taxonomy() -> Dict[str, dict]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def detect_domains(query: str) -> List[str]:
    taxonomy = _load_taxonomy()
    lowered = _normalize(query)
    detected = ["business_law"]
    for domain, meta in taxonomy.items():
        if domain == "business_law":
            continue
        if any(_normalize(keyword) in lowered for keyword in meta.get("keywords", [])):
            detected.append(domain)
    if "nuoc ngoai" in lowered and ("gop von" in lowered or "dau tu" in lowered):
        detected.append("investment_law")
    return list(dict.fromkeys(detected))


def route_query(query: str, seed_chunks: Iterable[Dict[str, object]] | None = None) -> Dict[str, object]:
    lowered = _normalize(query)
    domains = detect_domains(query)
    seed_chunks = list(seed_chunks or [])
    if any(chunk.get("metadata", {}).get("domain") for chunk in seed_chunks):
        for chunk in seed_chunks:
            domain = chunk.get("metadata", {}).get("domain")
            if domain and domain not in domains:
                domains.append(str(domain))

    route = SIMPLE_VECTOR
    reason = "Default simple retrieval."
    needs_parent = False
    needs_neighbor = False
    needs_graph = False
    needs_cross_domain = False

    if any(token in lowered for token in ["toan bo", "nhung viec gi", "can lam gi", "cac nghia vu", "can nhung van ban nao", "so sanh"]):
        route = MULTI_DOMAIN_COMPLEX
        reason = "Query asks for broad multi-step or multi-obligation guidance."
        needs_parent = True
        needs_neighbor = True
        needs_graph = True
        needs_cross_domain = len(domains) > 1
    elif len(domains) > 1 or any(token in lowered for token in ["bi phat", "xu phat", "nha dau tu nuoc ngoai", "nguoi nuoc ngoai", "fdi", "thue", "lao dong", "bhxh", "hop dong"]):
        route = CROSS_DOMAIN_CONTEXT
        reason = "Query touches business law plus a satellite legal domain."
        needs_parent = True
        needs_graph = True
        needs_cross_domain = True
    elif any(token in lowered for token in ["lien quan", "can cu", "theo quy dinh tai", "huong dan boi", "sua doi", "bo sung", "thay the", "het hieu luc", "con hieu luc", "ngoai le", "tru truong hop"]):
        route = LEGAL_GRAPH_CONTEXT
        reason = "Query asks for legal relationship or legal status expansion."
        needs_parent = True
        needs_graph = True
    elif any(token in lowered for token in ["dieu", "khoan", "diem", "doi tuong nao", "truong hop nao", "dieu kien", "quyen", "nghia vu", "ai khong duoc"]):
        route = PARENT_CONTEXT
        reason = "Query requires article/clause level grounding."
        needs_parent = True
    return {
        "route": route,
        "domains": domains,
        "needs_parent": needs_parent,
        "needs_neighbor": needs_neighbor,
        "needs_graph": needs_graph,
        "needs_cross_domain": needs_cross_domain,
        "reason": reason,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Route legal query to retrieval strategy.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    print(json.dumps(route_query(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
