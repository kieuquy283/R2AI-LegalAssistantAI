import sqlite3
import json
import time
from pathlib import Path
from tqdm import tqdm

def create_chunks_sqlite(jsonl_path: Path, db_path: Path):
    """Convert merged_chunks.jsonl to SQLite for fast random access."""
    if db_path.exists():
        print(f"SQLite DB already exists: {db_path}")
        return
    
    print(f"Creating SQLite DB from {jsonl_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            row_idx INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            doc_id TEXT,
            doc_title TEXT,
            doc_number TEXT,
            article TEXT,
            clause TEXT,
            citation TEXT,
            domain TEXT,
            source_url TEXT,
            content TEXT,
            priority INTEGER DEFAULT 0,
            metadata TEXT
        )
    """)
    
    # Create index on chunk_id for fast lookup
    cursor.execute("CREATE INDEX idx_chunk_id ON chunks(chunk_id)")
    cursor.execute("CREATE INDEX idx_doc_id ON chunks(doc_id)")
    
    # Count total lines
    total_lines = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    # Insert in batches
    batch = []
    batch_size = 1000
    inserted = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Inserting rows")):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            
            batch.append((
                i,
                str(row.get("chunk_id") or ""),
                str(row.get("doc_id") or ""),
                str(row.get("doc_title") or ""),
                str(row.get("doc_number") or ""),
                str(row.get("article") or ""),
                str(row.get("clause") or ""),
                str(row.get("citation") or ""),
                str(row.get("domain") or ""),
                str(row.get("source_url") or ""),
                str(row.get("content") or ""),
                int(row.get("priority") or 0),
                json.dumps(row, ensure_ascii=False)
            ))
            
            if len(batch) >= batch_size:
                cursor.executemany("""
                    INSERT INTO chunks 
                    (row_idx, chunk_id, doc_id, doc_title, doc_number, article, clause, citation, domain, source_url, content, priority, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                inserted += len(batch)
                batch = []
    
    if batch:
        cursor.executemany("""
            INSERT INTO chunks 
            (row_idx, chunk_id, doc_id, doc_title, doc_number, article, clause, citation, domain, source_url, content, priority, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        inserted += len(batch)
    
    conn.commit()
    conn.close()
    
    print(f"Created SQLite DB: {db_path}")
    print(f"Inserted {inserted} rows")
    print(f"DB size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    jsonl_path = Path("data/processed/merged_chunks.jsonl")
    db_path = Path("data/cache/chunks.db")
    
    t0 = time.perf_counter()
    create_chunks_sqlite(jsonl_path, db_path)
    t1 = time.perf_counter()
    print(f"Total time: {t1-t0:.1f}s")
