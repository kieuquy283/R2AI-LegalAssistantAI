"""Post-filter bad doc_keys, re-run for affected entries."""
import json
import re
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

p = re.compile(r"\b(\d+(?:/\d+)*(?:-[A-Z]+)*/[A-Z0-9À-ỴĂÂĐÊÔƠƯ\-]+)\b")

with open("data/processed/submission_parquet.json", "r", encoding="utf-8") as f:
    data = json.load(f)

to_rerun = []
clean_data = []

for d in data:
    docs_bad = [k for k in d["relevant_docs"] if not p.match(k.split("|")[0])]
    arts_bad = [k for k in d["relevant_articles"] if not p.match(k.split("|")[0])]

    clean_d = dict(d)
    clean_d["relevant_docs"] = [k for k in d["relevant_docs"] if k not in docs_bad]
    clean_d["relevant_articles"] = [k for k in d["relevant_articles"] if k not in arts_bad]

    if len(clean_d["relevant_docs"]) == 0 or len(clean_d["relevant_articles"]) == 0:
        to_rerun.append(d)
        clean_data.append(None)
    else:
        clean_data.append(clean_d)

print(f"Entries to re-run (0 docs or 0 arts): {len(to_rerun)}")
for d in to_rerun:
    nd = len(d["relevant_docs"])
    na = len(d["relevant_articles"])
    bad_d = sum(1 for k in d["relevant_docs"] if not p.match(k.split("|")[0]))
    bad_a = sum(1 for k in d["relevant_articles"] if not p.match(k.split("|")[0]))
    print(f"  {d['id']}: docs={nd}({bad_d} bad), arts={na}({bad_a} bad)")

if len(to_rerun) == 0:
    print("No entries need re-run. Writing clean file directly.")
    with open("data/processed/submission_parquet.json", "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False)
    print("Done.")
else:
    # Save temp checkpoint with clean data + re-run list
    out = []
    for d in clean_data:
        if d is not None:
            out.append(d)
    with open("data/processed/submission_parquet_clean_temp.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    with open("data/processed/submission_parquet_rerun_ids.json", "w", encoding="utf-8") as f:
        json.dump([d["id"] for d in to_rerun], f, ensure_ascii=False)
    print(f"Saved clean {len(out)} entries + {len(to_rerun)} rerun IDs.")
