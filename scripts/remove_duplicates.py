import os
import dotenv
from scripts.chroma_client import ChromaDBManager

dotenv.load_dotenv()

db = ChromaDBManager(db_path='./discord_data_db')
col = db.client.get_collection('messages')

data = col.get(include=['documents', 'metadatas'])

print(f"Total documents: {len(data['ids'])}")

seen = {}
duplicates = []

for i, (doc_id, metadata, content) in enumerate(zip(data['ids'], data['metadatas'], data['documents'])):
    key = f"{metadata.get('timestamp')}|{metadata.get('author')}|{content[:100]}"
    
    if key in seen:
        duplicates.append(doc_id)
        print(f"Duplicate found: {doc_id}")
        print(f"  Original: {seen[key]}")
        print(f"  Time: {metadata.get('timestamp')[:19]}")
        print(f"  Author: {metadata.get('author')}")
        print(f"  Content: {content[:80]}...")
        print()
    else:
        seen[key] = doc_id

print(f"\nTotal duplicates: {len(duplicates)}")

if duplicates:
    print(f"Deleting {len(duplicates)} duplicates...")
    col.delete(ids=duplicates)
    print("Done!")
else:
    print("No duplicates to delete.")
