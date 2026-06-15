import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from tqdm import tqdm

# Config
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
SOURCE_COLLECTION = "legal_chunks"
TARGET_COLLECTION = "legal_articles"
VECTOR_SIZE = 1024
BATCH_SIZE = 100

def main():
    print("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Check collections
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    print(f"Existing collections: {collection_names}")
    
    if SOURCE_COLLECTION not in collection_names:
        print(f"ERROR: Source collection '{SOURCE_COLLECTION}' not found!")
        return
    
    # Create target collection
    if TARGET_COLLECTION not in collection_names:
        print(f"Creating collection {TARGET_COLLECTION}...")
        client.create_collection(
            collection_name=TARGET_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
    
    # Get source count
    source_count = client.count(collection_name=SOURCE_COLLECTION)
    print(f"Source '{SOURCE_COLLECTION}' has {source_count.count} points")
    
    # Copy points with article-focused payload
    print(f"Copying to '{TARGET_COLLECTION}'...")
    offset = 0
    total_copied = 0
    
    while True:
        # Scroll through source collection
        results = client.scroll(
            collection_name=SOURCE_COLLECTION,
            offset=offset,
            limit=BATCH_SIZE,
            with_payload=True,
            with_vectors=True,
        )
        
        points = results[0]
        if not points:
            break
        
        # Transform to article-focused payload
        new_points = []
        for point in points:
            payload = point.payload
            
            # Extract article info from payload
            new_payload = {
                "article_id": str(point.id),
                "doc_id": str(payload.get("doc_id") or ""),
                "doc_title": str(payload.get("doc_title") or ""),
                "doc_number": str(payload.get("doc_number") or ""),
                "doc_type": str(payload.get("doc_type") or ""),
                "article": str(payload.get("article") or payload.get("chunk_id") or point.id),
                "article_title": str(payload.get("article_title") or payload.get("doc_title") or ""),
                "domain": str(payload.get("domain") or ""),
                "sector": str(payload.get("sector") or ""),
                "field": str(payload.get("field") or ""),
                "content": str(payload.get("content") or payload.get("text", ""))[:1000],
                "metadata": payload,
            }
            
            new_point = PointStruct(
                id=point.id,
                vector=point.vector,
                payload=new_payload
            )
            new_points.append(new_point)
        
        # Upsert to target
        client.upsert(collection_name=TARGET_COLLECTION, points=new_points)
        total_copied += len(new_points)
        
        # Update offset for next batch
        offset = points[-1].id + 1 if points else None
        if offset is None:
            break
        
        if total_copied % 1000 == 0:
            print(f"Copied {total_copied} articles...")
    
    # Verify
    target_count = client.count(collection_name=TARGET_COLLECTION)
    print(f"\nDone! Collection '{TARGET_COLLECTION}' now has {target_count.count} articles")
    print(f"Copied from {source_count.count} chunks in '{SOURCE_COLLECTION}'")
    
    # Test search
    print("\nTesting search...")
    # Get a sample point
    sample = client.scroll(
        collection_name=TARGET_COLLECTION,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )[0][0]
    
    print(f"Sample article: {sample.payload['article']}")
    print(f"Doc title: {sample.payload['doc_title'][:60]}")
    print(f"Content preview: {sample.payload['content'][:100]}")

if __name__ == "__main__":
    main()
