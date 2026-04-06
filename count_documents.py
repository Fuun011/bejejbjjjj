import chromadb
import os

def count_documents():
    """Count all documents in the ChromaDB database."""
    db_path = "./discord_data_db"
    
    if not os.path.exists(db_path):
        print(f"Database không tồn tại tại: {db_path}")
        return
    
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    
    print(f"Tổng số collections: {len(collections)}")
    print("-" * 40)
    
    total_docs = 0
    for collection in collections:
        count = collection.count()
        total_docs += count
        print(f"Collection '{collection.name}' (id: {collection.id}): {count} documents")
        
        # Xem sample documents
        if count > 0:
            sample = collection.peek(limit=3)
            print(f"  Sample IDs: {sample['ids'][:3] if sample['ids'] else 'N/A'}")
    
    print("-" * 40)
    print(f"Tổng số documents: {total_docs}")
    
    # Tính toán lý thuyết
    # Gemini embedding: 3072 dimensions × 4 bytes = 12,288 bytes/document
    print(f"\nƯớc tính dung lượng embedding: {total_docs * 12288 / 1024 / 1024:.2f} MB")
    print("Phần còn lại là metadata, index, và dữ liệu đã xóa (chưa vacuum)")

if __name__ == "__main__":
    count_documents()
