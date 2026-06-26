"""Clean anomalous doc_number in legal_parquet_v2:
- Delete "VĂN BẢN NÀY TRÙNG" (17 points)
- Normalize leading quotes/dots, "Số:" prefix, "Thông tư" prefix
- Normalize "Khongso" variants
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import re
from collections import Counter
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
col = "legal_parquet_v2"

BATCH = 500
DELETE_BATCH = 100

FIX_PATTERNS = [
    (re.compile(r"^'+"), ""),
    (re.compile(r"^\.+"), ""),
    (re.compile(r"^Số:\s*", re.IGNORECASE), ""),
    (re.compile(r"^số:\s*", re.IGNORECASE), ""),
    (re.compile(r"^Thông tư\s+", re.IGNORECASE), ""),
    (re.compile(r"^Thông tư số\s+", re.IGNORECASE), ""),
    (re.compile(r"^Nghị quyết số:\s*", re.IGNORECASE), ""),
    (re.compile(r"\.$"), ""),
]

DELETE_KEYWORDS = ["VĂN BẢN NÀY TRÙNG", "VĂN BẢN TRÙNG"]

def is_delete_doc(dn: str) -> bool:
    return any(kw in dn.upper() for kw in DELETE_KEYWORDS)

def fix_doc_number(dn: str) -> str | None:
    if is_delete_doc(dn):
        return "DELETE"
    fixed = dn
    for pattern, repl in FIX_PATTERNS:
        fixed = pattern.sub(repl, fixed)
    if fixed != dn:
        return fixed
    return None

# --- Scan ---
total = 0
to_delete = []
to_update = {}

offset = None
while True:
    records, next_offset = client.scroll(col, limit=BATCH, offset=offset, with_payload=True, with_vectors=False)
    if not records:
        break
    for r in records:
        total += 1
        dn = str(r.payload.get("doc_number", "") or "")
        result = fix_doc_number(dn)
        if result == "DELETE":
            to_delete.append(r.id)
        elif result is not None:
            to_update[r.id] = result
    if total % 10000 == 0:
        print(f"  Scanned {total}... delete={len(to_delete)} update={len(to_update)}")
    offset = next_offset
    if offset is None:
        break

print(f"\nScanned {total} total")
print(f"To delete: {len(to_delete)}")
print(f"To update: {len(to_update)}")

# --- Phase 2: Delete ---
if to_delete:
    for i in range(0, len(to_delete), DELETE_BATCH):
        batch = to_delete[i:i+DELETE_BATCH]
        client.delete(collection_name=col, points_selector=batch, wait=True)
    print(f"  Deleted {len(to_delete)} points")

# --- Phase 3: Update (one batch per distinct doc_number value) ---
if to_update:
    # Group by new doc_number for efficient batching
    by_value = {}
    for pid, new_dn in to_update.items():
        by_value.setdefault(new_dn, []).append(pid)
    updated = 0
    for new_dn, pids in by_value.items():
        for i in range(0, len(pids), DELETE_BATCH):
            batch_pids = pids[i:i+DELETE_BATCH]
            try:
                client.set_payload(
                    collection_name=col,
                    payload={"doc_number": new_dn},
                    points=batch_pids,
                    wait=True,
                )
                updated += len(batch_pids)
            except Exception as e:
                print(f"  Error updating batch: {e}")
    print(f"  Updated {updated} points")

# --- Phase 4: Fix tags ---
print("\nFixing tag_1/tag_2...")
offset = None
tag_fixes = 0
while True:
    records, next_offset = client.scroll(col, limit=BATCH, offset=offset, with_payload=True, with_vectors=False)
    if not records:
        break
    for r in records:
        pl = r.payload
        dn = str(pl.get("doc_number", "") or "")
        tag1 = str(pl.get("tag_1", "") or "")
        tag2 = str(pl.get("tag_2", "") or "")
        fix_payload = {}
        if tag1 and not tag1.startswith(dn) and dn not in tag1:
            parts = tag1.split("|", 1)
            if len(parts) == 2 and parts[1].strip():
                fix_payload["tag_1"] = f"{dn}|{parts[1].strip()}"
        if tag2 and not tag2.startswith(dn) and dn not in tag2:
            parts = tag2.split("|", 1)
            if len(parts) == 2 and parts[1].strip():
                fix_payload["tag_2"] = f"{dn}|{parts[1].strip()}"
        if fix_payload:
            try:
                client.set_payload(collection_name=col, payload=fix_payload, points=[r.id], wait=True)
                tag_fixes += 1
            except Exception as e:
                print(f"  Error fixing tag for point {r.id}: {e}")
    offset = next_offset
    if offset is None:
        break
print(f"  Fixed tags: {tag_fixes}")

info = client.get_collection(col)
print(f"\nCollection: {info.points_count} points (removed {total - info.points_count})")
