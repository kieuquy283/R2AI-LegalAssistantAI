import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
from tqdm import tqdm

# Config
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "legal_articles"
VECTOR_SIZE = 1024
BATCH_SIZE = 100

# Paths
CHUNKS_PATH = "data/embeddings/r2ai_expanded_corpus/expanded_chunks.jsonl"

def main():
    print("Loading chunks...")
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Connect to Qdrant
    print("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Check collection
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    print(f"Existing collections: {collection_names}")
    
    if COLLECTION_NAME not in collection_names:
        print(f"Creating collection {COLLECTION_NAME}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
    
    # Upsert in batches
    print("Upserting articles...")
    points = []
    for i, chunk in enumerate(tqdm(chunks)):
        # Create article ID from chunk_id
        chunk_id = str(chunk.get("chunk_id") or f"chunk_{i}")
        # Use hash of chunk_id for numeric ID
        article_id = hash(chunk_id) % (2**63)  # Ensure positive int64
        
        payload = {
            "article_id": chunk_id,
            "doc_id": str(chunk.get("doc_id") or ""),
            "doc_title": str(chunk.get("doc_title") or ""),
            "doc_number": str(chunk.get("doc_number") or ""),
            "doc_type": str(chunk.get("doc_type") or ""),
            "article": str(chunk.get("article") or chunk_id),  # Use chunk_id as article if no article
            "article_title": str(chunk.get("article_title") or chunk.get("doc_title") or ""),
            "domain": str(chunk.get("domain") or ""),
            "sector": str(chunk.get("sector") or ""),
            "field": str(chunk.get("field") or ""),
            "content": str(chunk.get("text") or chunk.get("content", ""))[:1000],
            "metadata": chunk,
        }
        
        # We don't have embeddings for chunks, so we'll create a dummy vector
        # or use zero vector. In production, you'd embed with BAAI/bge-m3.
        # For now, use random unit vector as placeholder.
        vector = np.random.randn(VECTOR_SIZE).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        point = PointStruct(
            id=int(article_id),
            vector=vector.tolist(),
            payload=payload
        )
        points.append(point)
        
        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
    
    # Upsert remaining
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    # Verify
    count = client.count(collection_name=COLLECTION_NAME)
    print(f"\nDone! Collection '{COLLECTION_NAME}' now has {count.count} articles")
    
    # Note: these are placeholder vectors. For real semantic search,
    # you need to embed chunks with BAAI/bge-m3 on Kaggle/Colab.
    print("\nNOTE: These are random placeholder vectors!")
    print("For real semantic search, embed chunks with BAAI/bge-m3 and upsert real embeddings.")

if __name__ == "__main__":
    main()
