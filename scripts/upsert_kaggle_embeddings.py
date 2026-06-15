import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
from tqdm import tqdm

# Config
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "legal_docs"
VECTOR_SIZE = 1024
BATCH_SIZE = 100

# Paths
EMBEDDINGS_PATH = "documents_embeddings.npz"
DOCS_PATH = "kaggle_data/documents.jsonl"

def main():
    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    doc_ids = data["doc_ids"]
    doc_titles = data["doc_titles"]
    doc_numbers = data["doc_numbers"]
    doc_types = data["doc_types"]
    issuers = data["issuers"]
    domains = data["domains"]
    effect_status = data["effect_status"]
    
    print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
    
    # Load documents for text
    docs_map = {}
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                docs_map[d["doc_id"]] = d
    
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
    print("Upserting documents...")
    points = []
    for i in range(len(embeddings)):
        doc_id_str = str(doc_ids[i])
        doc = docs_map.get(doc_id_str, {})
        
        payload = {
            "doc_id": doc_id_str,
            "doc_title": str(doc_titles[i]),
            "doc_number": str(doc_numbers[i]),
            "doc_type": str(doc_types[i]),
            "issuer": str(issuers[i]),
            "domain": str(domains[i]),
            "effect_status": str(effect_status[i]),
            "cleaned_text": doc.get("cleaned_text", "")[:2000],  # Truncate for payload
        }
        
        point = PointStruct(
            id=int(doc_id_str) if doc_id_str.isdigit() else i,
            vector=embeddings[i].tolist(),
            payload=payload
        )
        points.append(point)
        
        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
            if (i + 1) % 1000 == 0:
                print(f"Upserted {i + 1} documents...")
    
    # Upsert remaining
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    # Verify
    count = client.count(collection_name=COLLECTION_NAME)
    print(f"\nDone! Collection '{COLLECTION_NAME}' now has {count.count} documents")
    
    # Test search
    print("\nTesting search...")
    test_vector = embeddings[0].tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=test_vector,
        limit=3
    )
    for r in results:
        print(f"  Score: {r.score:.4f} | {r.payload['doc_title'][:60]}")

if __name__ == "__main__":
    main()
