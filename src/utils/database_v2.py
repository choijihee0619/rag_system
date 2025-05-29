"""
MongoDB 클라이언트 v2.0
CREATED [2024-12-19]: 새로운 컬렉션 구조 지원

새로운 기능:
- Document, Labels, QAPairs, Folder 컬렉션 지원
- folder_id 기반 관계형 쿼리
- 벡터 검색 지원 (text_embedding 필드)
- 계층적 폴더 구조 지원
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from typing import List, Dict, Any, Optional, Union
import datetime
from bson import ObjectId
import numpy as np
from src.utils.schemas import MongoSchemas

class MongoDBClientV2:
    """새로운 스키마를 지원하는 MongoDB 클라이언트"""
    
    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db = self.client.rag_system
        
        # 새 컬렉션들
        self.documents = self.db.Document
        self.labels = self.db.Labels
        self.qa_pairs = self.db.QAPairs
        self.folders = self.db.Folder
        
        # 스키마 정의
        self.schemas = MongoSchemas()
    
    # ==================== Folder 관련 메서드 ====================
    
    def create_folder(self, title: str, folder_type: str = "general", 
                     parent_folder_id: Optional[ObjectId] = None,
                     description: str = None, **kwargs) -> ObjectId:
        """새 폴더 생성"""
        folder_data = {
            "title": title,
            "description": description,
            "folder_type": folder_type,
            "parent_folder_id": parent_folder_id,
            "created_at": datetime.datetime.utcnow(),
            "last_accessed_at": datetime.datetime.utcnow(),
            "cover_image_url": kwargs.get("cover_image_url"),
            "metadata": kwargs.get("metadata", {})
        }
        
        result = self.folders.insert_one(folder_data)
        return result.inserted_id
    
    def get_folder(self, folder_id: ObjectId) -> Optional[Dict[str, Any]]:
        """폴더 조회"""
        return self.folders.find_one({"_id": folder_id})
    
    def list_folders(self, parent_folder_id: Optional[ObjectId] = None) -> List[Dict[str, Any]]:
        """폴더 목록 조회 (계층적)"""
        query = {"parent_folder_id": parent_folder_id}
        return list(self.folders.find(query).sort("created_at", DESCENDING))
    
    def update_folder_access_time(self, folder_id: ObjectId):
        """폴더 접근 시간 업데이트"""
        self.folders.update_one(
            {"_id": folder_id},
            {"$set": {"last_accessed_at": datetime.datetime.utcnow()}}
        )
    
    def delete_folder(self, folder_id: ObjectId, recursive: bool = False) -> bool:
        """폴더 삭제"""
        if recursive:
            # 하위 폴더와 모든 관련 문서도 삭제
            self._delete_folder_recursive(folder_id)
        else:
            # 빈 폴더만 삭제
            if self._is_folder_empty(folder_id):
                self.folders.delete_one({"_id": folder_id})
                return True
            else:
                return False
        return True
    
    def _delete_folder_recursive(self, folder_id: ObjectId):
        """폴더와 모든 하위 내용 재귀적 삭제"""
        # 하위 폴더들 찾아서 재귀 삭제
        child_folders = self.folders.find({"parent_folder_id": folder_id})
        for child in child_folders:
            self._delete_folder_recursive(child["_id"])
        
        # 이 폴더의 모든 문서, 라벨, QA 삭제
        self.documents.delete_many({"folder_id": folder_id})
        self.labels.delete_many({"folder_id": folder_id})
        self.qa_pairs.delete_many({"folder_id": folder_id})
        
        # 폴더 자체 삭제
        self.folders.delete_one({"_id": folder_id})
    
    def _is_folder_empty(self, folder_id: ObjectId) -> bool:
        """폴더가 비어있는지 확인"""
        has_documents = self.documents.count_documents({"folder_id": folder_id}) > 0
        has_subfolders = self.folders.count_documents({"parent_folder_id": folder_id}) > 0
        return not (has_documents or has_subfolders)
    
    # ==================== Document 관련 메서드 ====================
    
    def insert_document(self, folder_id: ObjectId, raw_text: str, 
                       chunk_sequence: Union[str, int], 
                       text_embedding: List[float] = None,
                       metadata: Dict[str, Any] = None) -> ObjectId:
        """Document 삽입"""
        document_data = {
            "folder_id": folder_id,
            "chunk_sequence": chunk_sequence,
            "raw_text": raw_text,
            "text_embedding": text_embedding or [],
            "metadata": metadata or {},
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        result = self.documents.insert_one(document_data)
        
        # 폴더 접근 시간 업데이트
        self.update_folder_access_time(folder_id)
        
        return result.inserted_id
    
    def get_document(self, document_id: ObjectId) -> Optional[Dict[str, Any]]:
        """Document 조회"""
        return self.documents.find_one({"_id": document_id})
    
    def get_documents_by_folder(self, folder_id: ObjectId, 
                               limit: int = None) -> List[Dict[str, Any]]:
        """폴더별 Document 목록 조회"""
        query = {"folder_id": folder_id}
        cursor = self.documents.find(query).sort("chunk_sequence", ASCENDING)
        
        if limit:
            cursor = cursor.limit(limit)
            
        return list(cursor)
    
    def search_documents_by_text(self, text_query: str, folder_id: ObjectId = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """텍스트 검색"""
        query = {"$text": {"$search": text_query}}
        
        if folder_id:
            query["folder_id"] = folder_id
        
        return list(self.documents.find(query).limit(limit))
    
    def update_document_embedding(self, document_id: ObjectId, 
                                 text_embedding: List[float]):
        """Document의 임베딩 업데이트"""
        self.documents.update_one(
            {"_id": document_id},
            {
                "$set": {
                    "text_embedding": text_embedding,
                    "updated_at": datetime.datetime.utcnow()
                }
            }
        )
    
    def vector_search(self, query_embedding: List[float], 
                     folder_id: ObjectId = None, k: int = 5) -> List[Dict[str, Any]]:
        """벡터 유사도 검색 (간단한 구현)"""
        # 실제 운영에서는 MongoDB Atlas Search나 전용 벡터 DB 사용 권장
        query = {}
        if folder_id:
            query["folder_id"] = folder_id
        
        # 임베딩이 있는 문서들만 조회
        query["text_embedding"] = {"$exists": True, "$ne": []}
        
        documents = list(self.documents.find(query))
        
        # 코사인 유사도 계산 (단순 구현)
        results = []
        query_norm = np.linalg.norm(query_embedding)
        
        for doc in documents:
            if doc.get("text_embedding"):
                doc_embedding = np.array(doc["text_embedding"])
                doc_norm = np.linalg.norm(doc_embedding)
                
                if doc_norm > 0 and query_norm > 0:
                    similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
                    results.append({
                        "document": doc,
                        "similarity": float(similarity)
                    })
        
        # 유사도 순으로 정렬하고 상위 k개 반환
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]
    
    # ==================== Labels 관련 메서드 ====================
    
    def insert_labels(self, document_id: ObjectId, folder_id: ObjectId,
                     main_topic: str, tags: List[str], category: str,
                     confidence: float = 0.8) -> ObjectId:
        """Labels 삽입"""
        label_data = {
            "document_id": document_id,
            "folder_id": folder_id,
            "main_topic": main_topic,
            "tags": tags,
            "category": category,
            "confidence": confidence,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        result = self.labels.insert_one(label_data)
        return result.inserted_id
    
    def get_labels_by_document(self, document_id: ObjectId) -> List[Dict[str, Any]]:
        """Document별 Labels 조회"""
        return list(self.labels.find({"document_id": document_id}))
    
    def search_by_tags(self, tags: List[str], folder_id: ObjectId = None) -> List[Dict[str, Any]]:
        """태그로 검색"""
        query = {"tags": {"$in": tags}}
        
        if folder_id:
            query["folder_id"] = folder_id
            
        return list(self.labels.find(query))
    
    def search_by_category(self, category: str, folder_id: ObjectId = None) -> List[Dict[str, Any]]:
        """카테고리로 검색"""
        query = {"category": category}
        
        if folder_id:
            query["folder_id"] = folder_id
            
        return list(self.labels.find(query))
    
    def get_popular_tags(self, folder_id: ObjectId = None, limit: int = 20) -> List[Dict[str, Any]]:
        """인기 태그 조회"""
        match_stage = {}
        if folder_id:
            match_stage["folder_id"] = folder_id
        
        pipeline = []
        if match_stage:
            pipeline.append({"$match": match_stage})
        
        pipeline.extend([
            {"$unwind": "$tags"},
            {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ])
        
        return list(self.labels.aggregate(pipeline))
    
    # ==================== QAPairs 관련 메서드 ====================
    
    def insert_qa_pair(self, document_id: ObjectId, folder_id: ObjectId,
                      question: str, answer: str, question_type: str = "general",
                      difficulty: str = "medium") -> ObjectId:
        """QAPair 삽입"""
        qa_data = {
            "document_id": document_id,
            "folder_id": folder_id,
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "difficulty": difficulty,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        result = self.qa_pairs.insert_one(qa_data)
        return result.inserted_id
    
    def get_qa_pairs_by_document(self, document_id: ObjectId) -> List[Dict[str, Any]]:
        """Document별 QA 쌍 조회"""
        return list(self.qa_pairs.find({"document_id": document_id}))
    
    def search_qa_pairs(self, query: str, folder_id: ObjectId = None,
                       question_type: str = None) -> List[Dict[str, Any]]:
        """QA 검색"""
        search_query = {
            "$or": [
                {"question": {"$regex": query, "$options": "i"}},
                {"answer": {"$regex": query, "$options": "i"}}
            ]
        }
        
        if folder_id:
            search_query["folder_id"] = folder_id
            
        if question_type:
            search_query["question_type"] = question_type
        
        return list(self.qa_pairs.find(search_query).limit(10))
    
    def get_qa_pairs_by_difficulty(self, difficulty: str, 
                                  folder_id: ObjectId = None) -> List[Dict[str, Any]]:
        """난이도별 QA 쌍 조회"""
        query = {"difficulty": difficulty}
        
        if folder_id:
            query["folder_id"] = folder_id
            
        return list(self.qa_pairs.find(query))
    
    # ==================== 통합 검색 메서드 ====================
    
    def hybrid_search(self, query: str, folder_id: ObjectId = None, 
                     k: int = 5) -> Dict[str, Any]:
        """하이브리드 검색 (텍스트 + QA)"""
        # 텍스트 검색
        text_results = self.search_documents_by_text(query, folder_id, limit=k)
        
        # QA 검색
        qa_results = self.search_qa_pairs(query, folder_id)
        
        # 태그 검색 (단어 단위)
        query_words = query.split()
        tag_results = self.search_by_tags(query_words, folder_id)
        
        return {
            "text_results": text_results,
            "qa_results": qa_results,
            "tag_results": tag_results,
            "total_results": len(text_results) + len(qa_results) + len(tag_results)
        }
    
    # ==================== 통계 및 유틸리티 ====================
    
    def get_folder_statistics(self, folder_id: ObjectId) -> Dict[str, Any]:
        """폴더 통계 정보"""
        document_count = self.documents.count_documents({"folder_id": folder_id})
        label_count = self.labels.count_documents({"folder_id": folder_id})
        qa_count = self.qa_pairs.count_documents({"folder_id": folder_id})
        
        # 최근 활동
        recent_documents = list(self.documents.find(
            {"folder_id": folder_id}
        ).sort("created_at", DESCENDING).limit(5))
        
        return {
            "folder_id": str(folder_id),
            "document_count": document_count,
            "label_count": label_count,
            "qa_count": qa_count,
            "recent_documents": recent_documents
        }
    
    def get_all_folder_stats(self) -> List[Dict[str, Any]]:
        """모든 폴더 통계"""
        folders = list(self.folders.find())
        stats = []
        
        for folder in folders:
            folder_stats = self.get_folder_statistics(folder["_id"])
            folder_stats.update({
                "folder_title": folder["title"],
                "folder_type": folder["folder_type"]
            })
            stats.append(folder_stats)
        
        return stats
    
    def close(self):
        """연결 종료"""
        self.client.close()
    
    # ==================== 호환성 메서드 (기존 코드 지원) ====================
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """기존 호환성: 모든 chunks(documents) 조회"""
        documents = list(self.documents.find())
        
        # 기존 형식으로 변환
        chunks = []
        for doc in documents:
            chunk = {
                "_id": doc["_id"],
                "chunk_id": doc.get("chunk_sequence", ""),
                "content": doc.get("raw_text", ""),
                "text": doc.get("raw_text", ""),
                "metadata": doc.get("metadata", {}),
                "filename": doc.get("metadata", {}).get("filename", "unknown"),
                "created_at": doc.get("created_at")
            }
            chunks.append(chunk)
        
        return chunks
    
    def find_by_labels(self, tags: List[str]) -> List[Dict[str, Any]]:
        """기존 호환성: 라벨로 검색"""
        return self.search_by_tags(tags) 