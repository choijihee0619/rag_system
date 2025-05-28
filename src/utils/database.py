from pymongo import MongoClient
from typing import List, Dict, Any
import datetime

class MongoDBClient:
    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db = self.client.rag_system
        self.chunks_collection = self.db.chunks
        self.labels_collection = self.db.labels
        self.qa_collection = self.db.qa_pairs
    
    def insert_chunk(self, chunk_data: Dict[str, Any]):
        """청크 데이터 삽입"""
        chunk_data["created_at"] = datetime.datetime.utcnow()
        return self.chunks_collection.insert_one(chunk_data)
    
    def insert_labels(self, label_data: Dict[str, Any]):
        """라벨 데이터 삽입"""
        label_data["created_at"] = datetime.datetime.utcnow()
        return self.labels_collection.insert_one(label_data)
    
    def insert_qa_pairs(self, qa_data: Dict[str, Any]):
        """QA 데이터 삽입"""
        qa_data["created_at"] = datetime.datetime.utcnow()
        return self.qa_collection.insert_one(qa_data)
    
    def find_by_labels(self, labels: List[str]) -> List[Dict[str, Any]]:
        """라벨로 검색"""
        query = {"labels.tags": {"$in": labels}}
        return list(self.labels_collection.find(query))
    
    def search_qa(self, query: str) -> List[Dict[str, Any]]:
        """QA 검색"""
        # 간단한 텍스트 검색 (실제로는 더 복잡한 검색 필요)
        regex_query = {"$regex": query, "$options": "i"}
        search_query = {
            "$or": [
                {"qa_pairs.question": regex_query},
                {"qa_pairs.answer": regex_query}
            ]
        }
        return list(self.qa_collection.find(search_query).limit(5))
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """모든 청크 조회"""
        return list(self.chunks_collection.find())
    
    def close(self):
        """연결 종료"""
        self.client.close()
