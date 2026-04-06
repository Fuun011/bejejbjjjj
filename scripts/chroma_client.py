import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any, Optional

class VietnameseEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Custom embedding function sử dụng local multilingual embedding model cho ChromaDB với batch processing.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", batch_size: int = 10):
        """
        Khởi tạo embedding function với model multilingual.
        
        Args:
            model_name (str): Tên model từ HuggingFace (mặc định: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
            batch_size (int): Số lượng texts xử lý trong mỗi batch để optimize performance.
        """
        print(f"INFO: Đang tải model embedding '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self._cache = {}
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"INFO: Model đã được tải. Dimension: {self.embedding_dim}")
    
    def __call__(self, input_texts: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = []
        
        try:
            for i in range(0, len(input_texts), self.batch_size):
                batch = input_texts[i:i + self.batch_size]
                batch_embeddings = []
                
                for text in batch:
                    cache_key = hash(text)
                    if cache_key in self._cache:
                        batch_embeddings.append(self._cache[cache_key])
                        continue
                    
                    # Sẽ được encode trong batch
                    batch_embeddings.append(None)
                
                # Encode những text chưa có cache
                texts_to_encode = [batch[j] for j, emb in enumerate(batch_embeddings) if emb is None]
                if texts_to_encode:
                    try:
                        encoded = self.model.encode(texts_to_encode, convert_to_list=True)
                        
                        # Gán lại embeddings
                        encoded_idx = 0
                        for j in range(len(batch_embeddings)):
                            if batch_embeddings[j] is None:
                                embedding = encoded[encoded_idx]
                                batch_embeddings[j] = embedding
                                
                                if len(self._cache) < 1000:
                                    cache_key = hash(batch[j])
                                    self._cache[cache_key] = embedding
                                
                                encoded_idx += 1
                    except Exception as e:
                        print(f"WARNING: Lỗi khi encode batch: {e}")
                        for j in range(len(batch_embeddings)):
                            if batch_embeddings[j] is None:
                                batch_embeddings[j] = [0.0] * self.embedding_dim
                
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"ERROR: Xảy ra lỗi nghiêm trọng khi tạo embedding: {e}")
            return [[0.0] * self.embedding_dim for _ in input_texts]
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        print("INFO: Embedding cache đã được xóa.")

class ChromaDBManager:
    """
    Một class quản lý ChromaDB mạnh mẽ và linh hoạt hơn với multilingual embedding.
    """
    def __init__(self, db_path: str = "./data.db", model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Khởi tạo client và cấu hình các dịch vụ cần thiết.

        Args:
            db_path (str): Đường dẫn đến thư mục lưu trữ database.
            model_name (str): Tên model embedding từ HuggingFace.
        """
        print("INFO: Đang khởi tạo ChromaDBManager...")
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = VietnameseEmbeddingFunction(model_name=model_name)
        self._collections: Dict[str, chromadb.Collection] = {}
        
        print(f"INFO: ChromaDBManager đã sẵn sàng. Dữ liệu được lưu tại: '{db_path}'")

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """
        Lấy hoặc tạo một collection và lưu vào cache để tái sử dụng.
        Đây là cách làm hiệu quả hơn thay vì gọi self.client.get... mỗi lần.

        Args:
            name (str): Tên của collection.

        Returns:
            chromadb.Collection: Đối tượng collection.
        """
        if name not in self._collections:
            print(f"INFO: Đang truy cập collection '{name}' lần đầu...")
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
        return self._collections[name]

    def add_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        metadatas: List[dict], 
        ids: List[str],
        skip_duplicates: bool = True
    ) -> int:
        """
        Thêm tài liệu vào một collection cụ thể với duplicate detection.

        Args:
            collection_name (str): Tên collection cần thêm vào.
            documents (List[str]): Danh sách nội dung các tài liệu.
            metadatas (List[dict]): Danh sách các metadata tương ứng.
            ids (List[str]): Danh sách các ID duy nhất tương ứng.
            skip_duplicates (bool): Bỏ qua documents với ID đã tồn tại.
            
        Returns:
            int: Số lượng documents thực sự được thêm vào DB.
        """
        if not documents:
            print("WARNING: Không có tài liệu nào để thêm.")
            return 0

        collection = self.get_or_create_collection(name=collection_name)
        
        if skip_duplicates:
            try:
                existing_data = collection.get(ids=ids, include=[])
                existing_ids = set(existing_data['ids']) if existing_data['ids'] else set()
                
                filtered_docs = []
                filtered_metas = []
                filtered_ids = []
                duplicates_count = 0
                
                for doc, meta, doc_id in zip(documents, metadatas, ids):
                    if doc_id not in existing_ids:
                        filtered_docs.append(doc)
                        filtered_metas.append(meta)
                        filtered_ids.append(doc_id)
                    else:
                        duplicates_count += 1
                
                if duplicates_count > 0:
                    print(f"INFO: Bỏ qua {duplicates_count} documents trùng lặp.")
                
                if not filtered_docs:
                    print("INFO: Tất cả documents đã tồn tại, không có gì để thêm.")
                    return 0
                
                documents = filtered_docs
                metadatas = filtered_metas
                ids = filtered_ids
                
            except Exception as e:
                print(f"WARNING: Không thể kiểm tra duplicates: {e}. Tiếp tục thêm...")
        
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return len(documents)
        except Exception as e:
            print(f"ERROR: Lỗi khi thêm documents: {e}")
            return self._add_documents_one_by_one(collection, documents, metadatas, ids)

    def search(
        self, 
        collection_name: str, 
        query_text: str, 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        min_relevance_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm thông tin trong collection bằng một câu truy vấn với filtering và ranking.

        Args:
            collection_name (str): Tên collection cần tìm kiếm.
            query_text (str): Câu hỏi hoặc nội dung cần tìm.
            n_results (int): Số lượng kết quả trả về tối đa.
            where (Optional[Dict]): Filter metadata (e.g., {"author": "username"}).
            where_document (Optional[Dict]): Filter document content.
            min_relevance_score (float): Điểm tương đồng tối thiểu (0-1). Distance sẽ được convert.

        Returns:
            List[Dict[str, Any]]: Danh sách các kết quả đã được định dạng và ranked.
        """
        collection = self.get_or_create_collection(name=collection_name)

        try:
            query_embedding = self.embedding_function.model.encode(query_text, convert_to_list=True)
        except Exception as e:
            print(f"LỖI: Không thể tạo embedding cho câu truy vấn: {e}")
            return []

        try:
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }
            
            if where:
                query_params["where"] = where
            if where_document:
                query_params["where_document"] = where_document
                
            results = collection.query(**query_params)
        except Exception as e:
            print(f"LỖI: Lỗi khi query collection: {e}")
            return []
        
        formatted_results = []
        if results and results.get('documents'):
            for i, doc_content in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                similarity_score = max(0.0, 1.0 - (distance / 2.0))
                
                if similarity_score < min_relevance_score:
                    continue
                    
                formatted_results.append({
                    'content': doc_content,
                    'metadata': results['metadatas'][0][i],
                    'distance': distance,
                    'similarity_score': similarity_score,
                    'rank': i + 1
                })
        
        return formatted_results
    
    def _add_documents_one_by_one(
        self, 
        collection: chromadb.Collection, 
        documents: List[str], 
        metadatas: List[dict], 
        ids: List[str]
    ) -> int:
        """
        Fallback method: thêm từng document riêng lẻ khi batch add fail.
        
        Args:
            collection: ChromaDB collection object.
            documents: List of document contents.
            metadatas: List of metadata dicts.
            ids: List of document IDs.
            
        Returns:
            int: Số lượng documents thành công.
        """
        success_count = 0
        failed_count = 0
        
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            try:
                collection.add(
                    documents=[doc],
                    metadatas=[meta],
                    ids=[doc_id]
                )
                success_count += 1
            except Exception as e:
                print(f"ERROR: Không thể thêm document ID '{doc_id}': {e}")
                failed_count += 1
        
        print(f"FALLBACK: Đã thêm {success_count}/{len(documents)} documents. Failed: {failed_count}")
        return success_count
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Lấy thống kê về một collection.
        
        Args:
            collection_name (str): Tên collection.
            
        Returns:
            Dict với thông tin: count, name, metadata
        """
        try:
            collection = self.get_or_create_collection(name=collection_name)
            count = collection.count()
            
            return {
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata if hasattr(collection, 'metadata') else {}
            }
        except Exception as e:
            print(f"ERROR: Không thể lấy stats cho collection '{collection_name}': {e}")
            return {"name": collection_name, "count": 0, "error": str(e)}
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Xóa một collection khỏi database.
        
        Args:
            collection_name (str): Tên collection cần xóa.
            
        Returns:
            bool: True nếu thành công.
        """
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
            print(f"SUCCESS: Đã xóa collection '{collection_name}'.")
            return True
        except Exception as e:
            print(f"ERROR: Không thể xóa collection '{collection_name}': {e}")
            return False
    
    def prepare_rag_context(
        self, 
        collection_name: str, 
        query_text: str, 
        max_tokens: int = 4000,
        n_results: int = 10,
        min_relevance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Chuẩn bị context cho RAG với quản lý token limit.
        
        Args:
            collection_name (str): Tên collection.
            query_text (str): Query từ user.
            max_tokens (int): Số tokens tối đa cho context (ước tính ~4 chars = 1 token).
            n_results (int): Số documents tối đa để retrieve.
            min_relevance (float): Điểm tương đồng tối thiểu.
            
        Returns:
            Dict với 'context', 'sources', 'total_tokens' (ước tính)
        """
        # Search for relevant documents
        results = self.search(
            collection_name=collection_name,
            query_text=query_text,
            n_results=n_results,
            min_relevance_score=min_relevance
        )
        
        if not results:
            return {
                "context": "",
                "sources": [],
                "total_tokens": 0,
                "message": "Không tìm thấy thông tin liên quan."
            }
        
        # Build context within token limit
        context_parts = []
        sources = []
        estimated_tokens = 0
        max_chars = max_tokens * 4  # Rough estimate: 4 chars ≈ 1 token
        
        for result in results:
            content = result['content']
            metadata = result['metadata']
            score = result['similarity_score']
            
            # Format context entry
            entry = f"[Score: {score:.2f}] {metadata.get('author', 'Unknown')}: {content}"
            entry_chars = len(entry)
            
            # Check if adding this would exceed limit
            if estimated_tokens * 4 + entry_chars > max_chars:
                break
            
            context_parts.append(entry)
            sources.append({
                "author": metadata.get('author', 'Unknown'),
                "timestamp": metadata.get('timestamp', 'Unknown'),
                "score": score
            })
            estimated_tokens += entry_chars // 4
        
        context = "\n\n".join(context_parts)
        
        return {
            "context": context,
            "sources": sources,
            "total_tokens": estimated_tokens,
            "documents_used": len(sources),
            "documents_found": len(results)
        }