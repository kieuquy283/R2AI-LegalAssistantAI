"""Run full 2000-evaluation with RRF 150 -> heuristic 20 -> API Qwen3 dynamic filter.
Logs per-question ID + total time for monitoring. Uses .env config (no hardcoded overrides).
"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, __file__.rsplit("\\", 2)[0])
from dotenv import load_dotenv
load_dotenv()

from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.qa_pipeline import LegalQAPipeline

# Load questions
with open("data/evaluation/r2ai_stage1_questions.jsonl", encoding="utf-8") as f:
    all_q = [json.loads(line) for line in f if line.strip()]

out_path = Path("data/processed/submission_parquet.json")
temp_path = Path("data/evaluation/r2ai_stage1_retrieval_temp.jsonl")

processed = set()
if temp_path.exists():
    with open(temp_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    processed.add(json.loads(line)["id"])
                except Exception:
                    pass
    print(f"Resume: {len(processed)} done", flush=True)

_DOC_RE = __import__("re").compile(r"\b(\d+(?:/\d+)*(?:-[A-Z]+)*/[A-Z0-9À-ỴĂÂĐÊÔƠƯ\-]+)\b")

RetrievalPipeline.reset_singleton()
LegalQAPipeline.reset_singleton()
p = LegalQAPipeline()

f_out = open(temp_path, "a", encoding="utf-8")
t0 = time.perf_counter()
done = err = 0

for q in all_q:
    if q["id"] in processed:
        continue
    q_t0 = time.perf_counter()
    try:
        result = p.answer(q["question"])
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

        row = {
            "id": q["id"],
            "question": q["question"],
            "answer": "",
            "relevant_docs": list(docs_seen.keys()),
            "relevant_articles": list(arts_seen.keys()),
        }
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        f_out.flush()
        done += 1
    except Exception as e:
        err += 1
        row = {"id": q["id"], "question": q["question"], "answer": "",
               "relevant_docs": [], "relevant_articles": []}
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        f_out.flush()
        print(f"  [{q['id']}] ERROR: {e}", flush=True)

    q_elapsed = time.perf_counter() - q_t0
    total_elapsed = time.perf_counter() - t0
    print(f"  [{q['id']}] {q_elapsed:.1f}s | {done}/{len(all_q)} err={err} total={total_elapsed:.0f}s", flush=True)

f_out.close()

# Build final sorted output
final = []
with open(temp_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            final.append(json.loads(line))
final.sort(key=lambda x: x["id"])
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
temp_path.unlink(missing_ok=True)
print(f"\nOutput: {out_path} ({len(final)} entries, {time.perf_counter()-t0:.0f}s)", flush=True)
