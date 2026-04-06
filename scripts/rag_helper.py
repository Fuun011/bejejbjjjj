"""
RAG Helper Module - Hybrid Search với Temporal Query Parsing
"""
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from scripts.chroma_client import ChromaDBManager

def get_local_timezone():
    offset = datetime.now().astimezone().utcoffset()
    return timezone(offset)

LOCAL_TZ = get_local_timezone()


class TemporalQueryParser:
    
    @staticmethod
    def parse(query: str) -> Optional[Dict[str, datetime]]:
        query_lower = query.lower()
        now = datetime.now(LOCAL_TZ)
        
        if any(word in query_lower for word in ["hôm qua", "hqua", "yesterday"]):
            target_date = now - timedelta(days=1)
            return {
                "date_start": target_date.replace(hour=0, minute=0, second=0, microsecond=0),
                "date_end": target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        
        if any(word in query_lower for word in ["hôm nay", "hnay", "today"]):
            return {
                "date_start": now.replace(hour=0, minute=0, second=0, microsecond=0),
                "date_end": now.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        
        # Hôm kia
        if any(word in query_lower for word in ["hôm kia", "hkia"]):
            target_date = now - timedelta(days=2)
            return {
                "date_start": target_date.replace(hour=0, minute=0, second=0, microsecond=0),
                "date_end": target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        
        # Tuần này
        if any(word in query_lower for word in ["tuần này", "this week"]):
            week_start = now - timedelta(days=now.weekday())
            return {
                "date_start": week_start.replace(hour=0, minute=0, second=0, microsecond=0),
                "date_end": now.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        
        # Tuần trước
        if any(word in query_lower for word in ["tuần trước", "tuần trước", "last week"]):
            week_start = now - timedelta(days=now.weekday() + 7)
            week_end = week_start + timedelta(days=6)
            return {
                "date_start": week_start.replace(hour=0, minute=0, second=0, microsecond=0),
                "date_end": week_end.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        
        # Tháng này
        if any(word in query_lower for word in ["tháng này", "this month"]):
            month_start = now.replace(day=1)
            return {
                "date_start": month_start.replace(hour=0, minute=0, second=0, microsecond=0),
                "date_end": now.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        
        # X ngày trước (pattern: "3 ngày trước")
        pattern = r'(\d+)\s*(ngày|ngay|day)s?\s*(trước|truoc|ago|trc)'
        match = re.search(pattern, query_lower)
        if match:
            days = int(match.group(1))
            target_date = now - timedelta(days=days)
            return {
                "date_start": target_date.replace(hour=0, minute=0, second=0, microsecond=0),
                "date_end": target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        
        return None
    
    @staticmethod
    def remove_temporal_keywords(query: str) -> str:
        """
        Xóa temporal keywords khỏi query để semantic search tốt hơn.
        
        Args:
            query: Original query
            
        Returns:
            Cleaned query without temporal keywords
        """
        temporal_keywords = [
            r'\bhôm qua\b', r'\bhqua\b', r'\byesterday\b',
            r'\bhôm nay\b', r'\bhnay\b', r'\btoday\b',
            r'\bhôm kia\b', r'\bhkia\b',
            r'\btuần này\b', r'\bthis week\b',
            r'\btuần trước\b', r'\blast week\b',
            r'\btháng này\b', r'\bthis month\b',
            r'\d+\s*(ngày|ngay|day)s?\s*(trước|truoc|ago)',
        ]
        
        cleaned = query
        for pattern in temporal_keywords:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()


class RAGHelper:
    """RAG Helper với hybrid search và re-ranking."""
    
    def __init__(self, db_manager: ChromaDBManager):
        """
        Initialize RAG Helper.
        
        Args:
            db_manager: ChromaDBManager instance
        """
        self.db_manager = db_manager
        self.temporal_parser = TemporalQueryParser()
    
    @staticmethod
    def _parse_timestamp_safe(timestamp_str: str) -> datetime:
        """
        Parse timestamp string và ensure timezone awareness.
        Support cả format cũ (YYYY-MM-DD HH:MM:SS) và mới (ISO with timezone).
        
        Args:
            timestamp_str: Timestamp string
            
        Returns:
            Timezone-aware datetime object
        """
        try:
            # Try ISO format with timezone first
            dt = datetime.fromisoformat(timestamp_str)
        except ValueError:
            # Fallback: parse format cũ "YYYY-MM-DD HH:MM:SS" (không có timezone)
            try:
                dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # Last resort: try removing microseconds
                dt = datetime.fromisoformat(timestamp_str.split('.')[0])
        
        # Nếu naive (không có timezone), thêm UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    
    def hybrid_search(
        self,
        query: str,
        collection_name: str = "messages",
        n_results: int = 10,
        min_relevance: float = 0.3,
        boost_recent: bool = True,
        recency_weight: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search
        Args:
            query: User query
            collection_name: Collection to search
            n_results: Max results to return
            min_relevance: Minimum similarity score
            boost_recent: Whether to boost recent messages
            recency_weight: Weight for recency in final score (0-1)
            
        Returns:
            List of search results with final_score
        """
        time_filter = self.temporal_parser.parse(query)
        
        clean_query = self.temporal_parser.remove_temporal_keywords(query)
        if not clean_query:
            clean_query = query  # Fallback nếu query chỉ toàn temporal keywords
        
        if time_filter:
            if len(clean_query.split()) <= 3:
                fetch_size = n_results * 50
            else:
                fetch_size = n_results * 20
        else:
            fetch_size = n_results * 5 
        
        results = self.db_manager.search(
            collection_name=collection_name,
            query_text=clean_query,
            n_results=fetch_size,
            min_relevance_score=min_relevance
        )
        
        if not results:
            return []
        
        if time_filter:
            date_start = time_filter['date_start']
            date_end = time_filter['date_end']
            
            filtered_results = []
            
            for idx, result in enumerate(results):
                try:
                    timestamp = self._parse_timestamp_safe(result['metadata']['timestamp'])
                    
                    if date_start <= timestamp <= date_end:
                        filtered_results.append(result)
                except (ValueError, KeyError) as e:
                    continue
            
            results = filtered_results
        
        if not results:
            return []
        
        if boost_recent:
            now = datetime.now(LOCAL_TZ)
            for result in results:
                timestamp = self._parse_timestamp_safe(result['metadata']['timestamp'])
                age_hours = (now - timestamp).total_seconds() / 3600
                
                recency_score = max(0.0, 1.0 / (1.0 + age_hours / 24))
                
                semantic_weight = 1.0 - recency_weight
                result['recency_score'] = recency_score
                result['final_score'] = (
                    result['similarity_score'] * semantic_weight +
                    recency_score * recency_weight
                )
            
            results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        return results[:n_results]
    
    def prepare_rag_prompt(
        self,
        query: str,
        max_context_tokens: int = 4000,
        n_results: int = 5,
        include_metadata: bool = True,
        include_nearby_messages: bool = True,
        nearby_time_window_minutes: int = 2
    ) -> Dict[str, Any]:
        """
        Chuẩn bị prompt cho RAG với context từ hybrid search.
        
        Args:
            query: User query
            max_context_tokens: Max tokens for context
            n_results: Number of documents to retrieve
            include_metadata: Include author and timestamp in context
            include_nearby_messages: Include messages within time window of top result
            nearby_time_window_minutes: Time window in minutes for nearby messages
            
        Returns:
            Dict với 'context', 'prompt', 'sources'
        """
        results = self.hybrid_search(
            query=query,
            n_results=n_results,
            boost_recent=True
        )
        
        if not results:
            return {
                "context": "Không tìm thấy thông tin liên quan.",
                "prompt": query,
                "sources": [],
                "documents_used": 0
            }
        
        all_results = results.copy()
        if include_nearby_messages and results:
            try:
                nearby_filtered = []
                seen_message_ids = {r['metadata'].get('message_id') for r in results}
                
                top_k = min(5, len(results))
                
                for rank_idx, top_result in enumerate(results[:top_k], 1):
                    top_timestamp = self._parse_timestamp_safe(top_result['metadata']['timestamp'])
                    
                    time_start = top_timestamp - timedelta(minutes=nearby_time_window_minutes)
                    time_end = top_timestamp + timedelta(minutes=nearby_time_window_minutes)
                    
                    nearby_results = self.db_manager.search(
                        collection_name="messages",
                        query_text=query,
                        n_results=30,
                        min_relevance_score=0.0
                    )
                    
                    for result in nearby_results:
                        try:
                            timestamp = self._parse_timestamp_safe(result['metadata']['timestamp'])
                            message_id = result['metadata'].get('message_id')
                            
                            if time_start <= timestamp <= time_end and message_id not in seen_message_ids:
                                nearby_filtered.append(result)
                                seen_message_ids.add(message_id)
                        except (ValueError, KeyError):
                            continue
                
                all_results = results + nearby_filtered
                all_results = sorted(all_results, key=lambda x: x['metadata']['timestamp'])
                
            except Exception as e:
                all_results = results
        
        context_parts = []
        sources = []
        estimated_tokens = 0
        max_chars = max_context_tokens * 4
        
        for i, result in enumerate(all_results, 1):
            content = result['content']
            metadata = result['metadata']
            
            if include_metadata:
                author = metadata.get('author', 'Unknown')
                timestamp = metadata.get('timestamp', 'Unknown')
                try:
                    dt = self._parse_timestamp_safe(timestamp)
                    time_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    time_str = timestamp
                
                entry = f"[{i}] {author} ({time_str}): {content}"
            else:
                entry = f"[{i}] {content}"
            
            entry_chars = len(entry)
            
            if estimated_tokens * 4 + entry_chars > max_chars:
                break
            
            context_parts.append(entry)
            sources.append({
                "author": metadata.get('author', 'Unknown'),
                "timestamp": metadata.get('timestamp', 'Unknown'),
                "similarity_score": result.get('similarity_score', 0),
                "final_score": result.get('final_score', 0)
            })
            estimated_tokens += entry_chars // 4
        
        context = "\n\n".join(context_parts)
        
        rag_prompt = f"""Dựa trên ngữ cảnh sau từ lịch sử chat:

{context}

Câu hỏi: {query}

Hãy trả lời dựa trên thông tin trong ngữ cảnh trên. Nếu không tìm thấy thông tin liên quan, hãy nói rằng bạn không tìm thấy."""
        
        return {
            "context": context,
            "prompt": rag_prompt,
            "sources": sources,
            "documents_used": len(sources),
            "estimated_tokens": estimated_tokens
        }
