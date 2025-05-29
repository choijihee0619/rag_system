"""
MongoDB 클라이언트 (호환성 레이어)
MODIFIED [2024-12-19]: 새로운 v2 스키마 지원하면서 기존 코드 호환성 유지
"""

from pymongo import MongoClient
from typing import List, Dict, Any
import datetime
from src.utils.database_v2 import MongoDBClientV2
from bson import ObjectId

class MongoDBClient:
    """기존 코드 호환성을 위한 래퍼 클래스"""
    
    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db = self.client.rag_system
        
        # 기존 컬렉션 (호환성을 위해 유지)
        self.chunks_collection = self.db.chunks
        self.labels_collection = self.db.labels
        self.qa_collection = self.db.qa_pairs
        
        # 새 v2 클라이언트
        self.v2 = MongoDBClientV2(uri)
        
        # 기본 폴더 ID 캐시
        self._default_folder_id = None
    
    def _get_default_folder_id(self) -> ObjectId:
        """기본 폴더 ID를 가져오거나 생성"""
        if self._default_folder_id is None:
            # 기존 기본 폴더 찾기
            default_folder = self.v2.folders.find_one({"folder_type": "general"})
            if default_folder:
                self._default_folder_id = default_folder["_id"]
            else:
                # 기본 폴더 생성
                self._default_folder_id = self.v2.create_folder(
                    title="기본 폴더",
                    folder_type="general",
                    description="호환성 레이어에서 생성된 기본 폴더"
                )
        return self._default_folder_id
    
    def insert_chunk(self, chunk_data: Dict[str, Any]):
        """청크 데이터 삽입 (새 스키마로 저장)"""
        folder_id = self._get_default_folder_id()
        
        # v2 형식으로 변환
        document_id = self.v2.insert_document(
            folder_id=folder_id,
            raw_text=chunk_data.get("content", chunk_data.get("text", "")),
            chunk_sequence=chunk_data.get("chunk_id", f"chunk_{datetime.datetime.now().timestamp()}"),
            text_embedding=chunk_data.get("text_embedding", []),
            metadata=chunk_data.get("metadata", {})
        )
        
        # 호환성을 위해 ObjectId를 포함한 결과 반환
        return type('MockResult', (), {'inserted_id': document_id})()
    
    def insert_labels(self, label_data: Dict[str, Any]):
        """라벨 데이터 삽입 (새 스키마로 저장)"""
        folder_id = self._get_default_folder_id()
        
        # chunk_id로 document_id 찾기
        chunk_id = label_data.get("chunk_id", "")
        document = self.v2.documents.find_one({"chunk_sequence": chunk_id})
        
        if not document:
            # 문서가 없으면 생성
            document_id = self.v2.insert_document(
                folder_id=folder_id,
                raw_text="",
                chunk_sequence=chunk_id,
                metadata={"auto_created": True}
            )
        else:
            document_id = document["_id"]
        
        # 라벨 구조 평탄화
        labels = label_data.get("labels", {})
        
        label_id = self.v2.insert_labels(
            document_id=document_id,
            folder_id=folder_id,
            main_topic=labels.get("main_topic", ""),
            tags=labels.get("tags", []),
            category=labels.get("category", "general"),
            confidence=0.8
        )
        
        return type('MockResult', (), {'inserted_id': label_id})()
    
    def insert_qa_pairs(self, qa_data: Dict[str, Any]):
        """QA 데이터 삽입 (새 스키마로 저장)"""
        folder_id = self._get_default_folder_id()
        
        # chunk_id로 document_id 찾기
        chunk_id = qa_data.get("chunk_id", "")
        document = self.v2.documents.find_one({"chunk_sequence": chunk_id})
        
        if not document:
            # 문서가 없으면 생성
            document_id = self.v2.insert_document(
                folder_id=folder_id,
                raw_text="",
                chunk_sequence=chunk_id,
                metadata={"auto_created": True}
            )
        else:
            document_id = document["_id"]
        
        # QA 쌍들을 개별 문서로 저장
        qa_pairs = qa_data.get("qa_pairs", [])
        inserted_ids = []
        
        for qa_pair in qa_pairs:
            if isinstance(qa_pair, dict) and "question" in qa_pair and "answer" in qa_pair:
                qa_id = self.v2.insert_qa_pair(
                    document_id=document_id,
                    folder_id=folder_id,
                    question=qa_pair["question"],
                    answer=qa_pair["answer"],
                    question_type="general",
                    difficulty="medium"
                )
                inserted_ids.append(qa_id)
        
        # 첫 번째 ID 반환 (호환성)
        return type('MockResult', (), {'inserted_id': inserted_ids[0] if inserted_ids else None})()
    
    def find_by_labels(self, labels: List[str]) -> List[Dict[str, Any]]:
        """라벨로 검색 (v2로 위임)"""
        results = self.v2.search_by_tags(labels)
        
        # 기존 형식으로 변환
        converted_results = []
        for result in results:
            converted = {
                "_id": result["_id"],
                "chunk_id": result.get("document_id"),  # document_id를 chunk_id로 매핑
                "labels": {
                    "main_topic": result.get("main_topic", ""),
                    "tags": result.get("tags", []),
                    "category": result.get("category", "general")
                },
                "created_at": result.get("created_at")
            }
            converted_results.append(converted)
        
        return converted_results
    
    def search_qa(self, query: str) -> List[Dict[str, Any]]:
        """QA 검색 (v2로 위임)"""
        results = self.v2.search_qa_pairs(query)
        
        # 기존 형식으로 변환 (qa_pairs 배열 형태로)
        grouped_results = {}
        
        for result in results:
            document_id = str(result.get("document_id", ""))
            
            if document_id not in grouped_results:
                grouped_results[document_id] = {
                    "_id": result["_id"],
                    "chunk_id": document_id,
                    "qa_pairs": [],
                    "created_at": result.get("created_at")
                }
            
            grouped_results[document_id]["qa_pairs"].append({
                "question": result.get("question", ""),
                "answer": result.get("answer", "")
            })
        
        return list(grouped_results.values())[:5]  # 최대 5개 결과
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """모든 청크 조회 (v2로 위임)"""
        return self.v2.get_all_chunks()
    
    def close(self):
        """연결 종료"""
        self.v2.close()
        self.client.close()
    
    # ==================== 새로운 v2 기능 직접 노출 ====================
    
    def create_folder(self, title: str, **kwargs) -> ObjectId:
        """폴더 생성"""
        return self.v2.create_folder(title, **kwargs)
    
    def get_folder_statistics(self, folder_id: ObjectId) -> Dict[str, Any]:
        """폴더 통계"""
        return self.v2.get_folder_statistics(folder_id)
    
    def hybrid_search(self, query: str, folder_id: ObjectId = None, k: int = 5) -> Dict[str, Any]:
        """하이브리드 검색"""
        return self.v2.hybrid_search(query, folder_id, k)
    
    def vector_search(self, query_embedding: List[float], folder_id: ObjectId = None, k: int = 5) -> List[Dict[str, Any]]:
        """벡터 검색"""
        return self.v2.vector_search(query_embedding, folder_id, k)
