"""Re-run pipeline for 3 empty entries and merge result."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

import re
_DOC_RE = re.compile(r"\b(\d+(?:/\d+)*(?:-[A-Z]+)*/[A-Z0-9À-ỴĂÂĐÊÔƠƯ\-]+)\b")

# Load questions
with open("data/evaluation/r2ai_stage1_questions.jsonl", encoding="utf-8") as f:
    all_q = [json.loads(line) for line in f if line.strip()]
q_map = {q["id"]: q["question"] for q in all_q}

# IDs to rerun
rerun_ids = {1484, 1485, 1486}

# Load current clean data (1997 entries)
with open("data/processed/submission_parquet.json", encoding="utf-8") as f:
    clean_data = json.load(f)
existing_ids = {d["id"] for d in clean_data}
print(f"Existing entries: {len(clean_data)}")
print(f"Missing IDs: {rerun_ids - existing_ids}")

from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.qa_pipeline import LegalQAPipeline

RetrievalPipeline.reset_singleton()
LegalQAPipeline.reset_singleton()
p = LegalQAPipeline()

t0 = time.perf_counter()
new_rows = []
for qid in sorted(rerun_ids):
    question = q_map.get(qid, "")
    qt0 = time.perf_counter()
    try:
        result = p.answer(question)
        ctx = result.get("final_contexts") or result.get("raw_final_contexts") or []
        docs_seen = {}
        arts_seen = {}
        for c in ctx:
            meta = c.get("metadata") or {}
            raw_dn = str(c.get("doc_number") or meta.get("doc_number") or "")
            raw_dt = str(c.get("doc_title") or meta.get("doc_title") or "")
            art = str(c.get("article") or meta.get("article") or "")
            dn, dt = raw_dn, raw_dt
            if " > " in raw_dn or any(kw in raw_dn for kw in ["Nghị định", "Luật", "Thông tư"]):
                m = _DOC_RE.search(raw_dt or raw_dn)
                if m:
                    dn = m.group(1)
                clean = raw_dt.split(" > ")[0].split(" | ")[0].strip()
                if clean:
                    dt = clean
            if dt.startswith(dn + "|") or dt.startswith(dn + " "):
                dt = dt[len(dn):].lstrip("| ")
            dk = f"{dn}|{dt}"
            if dk and dk not in docs_seen:
                docs_seen[dk] = True
            ak = f"{dn}|{dt}|{art}" if art else ""
            if ak and ak not in arts_seen:
                arts_seen[ak] = True
        row = {"id": qid, "question": question, "answer": "",
               "relevant_docs": list(docs_seen.keys()), "relevant_articles": list(arts_seen.keys())}
    except Exception as e:
        row = {"id": qid, "question": question, "answer": "",
               "relevant_docs": [], "relevant_articles": []}
        print(f"  [{qid}] ERROR: {e}", flush=True)
    new_rows.append(row)
    print(f"  [{qid}] {len(row['relevant_docs'])} docs, {len(row['relevant_articles'])} arts | {time.perf_counter()-qt0:.1f}s", flush=True)

# Merge
clean_data.extend(new_rows)
clean_data.sort(key=lambda x: x["id"])

out_path = "data/processed/submission_parquet.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=2)

print(f"\nFinal: {len(clean_data)} entries -> {out_path} | {time.perf_counter()-t0:.0f}s")

# Verify
p2 = re.compile(r"\b(\d+(?:/\d+)*(?:-[A-Z]+)*/[A-Z0-9À-ỴĂÂĐÊÔƠƯ\-]+)\b")
total_docs = total_arts = bad_d = bad_a = 0
for d in clean_data:
    for k in d["relevant_docs"]:
        total_docs += 1
        if not p2.match(k.split("|")[0]):
            bad_d += 1
    for k in d["relevant_articles"]:
        total_arts += 1
        if not p2.match(k.split("|")[0]):
            bad_a += 1
empty = sum(1 for d in clean_data if not d["relevant_docs"])
print(f"Verify: {len(clean_data)} entries, bad docs={bad_d}/{total_docs}, bad arts={bad_a}/{total_arts}, empty={empty}")
