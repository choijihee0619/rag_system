"""
MongoDB 컬렉션 스키마 정의
MODIFIED [2024-12-19]: 새로운 컬렉션 구조로 완전 재설계
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from bson import ObjectId
import pymongo

class MongoSchemas:
    """MongoDB 컬렉션 스키마 정의"""
    
    @staticmethod
    def get_document_schema() -> Dict[str, Any]:
        """Document 컬렉션 스키마 (기존 chunks)"""
        return {
            "_id": ObjectId,  # MongoDB 자동 생성
            "folder_id": ObjectId,  # 폴더 참조 (필수)
            "chunk_sequence": int,  # 기존 chunk_id를 대체
            "raw_text": str,  # 기존 content → raw_text
            "text_embedding": List[float],  # 벡터 DB에서 이관
            "metadata": Dict[str, Any],  # 기존 metadata 유지
            "created_at": datetime,  # 기존 유지
            "updated_at": datetime  # 신규 추가
        }
    
    @staticmethod
    def get_labels_schema() -> Dict[str, Any]:
        """Labels 컬렉션 스키마 (기존 labels)"""
        return {
            "_id": ObjectId,  # 자동 생성
            "document_id": ObjectId,  # chunk_id → document_id
            "folder_id": ObjectId,  # 신규 추가
            "main_topic": str,  # labels.main_topic에서 추출
            "tags": List[str],  # labels.tags에서 추출
            "category": str,  # labels.category에서 추출
            "confidence": float,  # 신규 추가 (라벨링 신뢰도)
            "created_at": datetime,  # 기존 유지
            "updated_at": datetime  # 신규 추가
        }
    
    @staticmethod
    def get_qa_pairs_schema() -> Dict[str, Any]:
        """QAPairs 컬렉션 스키마 (기존 qa_pairs)"""
        return {
            "_id": ObjectId,  # 자동 생성
            "document_id": ObjectId,  # chunk_id → document_id
            "folder_id": ObjectId,  # 신규 추가
            "question": str,  # qa_pairs 배열에서 개별 문서로 분리
            "answer": str,  # qa_pairs 배열에서 개별 문서로 분리
            "question_type": str,  # 신규 추가 (질문 유형)
            "difficulty": str,  # 신규 추가 (난이도: easy, medium, hard)
            "created_at": datetime,  # 기존 유지
            "updated_at": datetime  # 신규 추가
        }
    
    @staticmethod
    def get_folder_schema() -> Dict[str, Any]:
        """Folder 컬렉션 스키마 (신규 생성)"""
        return {
            "_id": ObjectId,  # 자동 생성
            "title": str,  # 폴더 제목
            "description": Optional[str],  # 폴더 설명
            "folder_type": str,  # 폴더 유형 (general, project, archive 등)
            "parent_folder_id": Optional[ObjectId],  # 상위 폴더 (계층 구조)
            "created_at": datetime,
            "last_accessed_at": datetime,
            "cover_image_url": Optional[str],  # null 허용
            "metadata": Dict[str, Any]  # 추가 메타데이터
        }
    
    @staticmethod
    def get_indexes() -> Dict[str, List[tuple]]:
        """각 컬렉션별 인덱스 정의"""
        return {
            "Document": [
                ("folder_id", pymongo.ASCENDING),
                ("chunk_sequence", pymongo.ASCENDING),
                ("created_at", pymongo.DESCENDING),
                ([("raw_text", "text")]),  # 텍스트 검색용
                ("text_embedding", "2dsphere")  # 벡터 검색용 (MongoDB Atlas Search)
            ],
            "Labels": [
                ("document_id", pymongo.ASCENDING),
                ("folder_id", pymongo.ASCENDING),
                ("main_topic", pymongo.ASCENDING),
                ("tags", pymongo.ASCENDING),
                ("category", pymongo.ASCENDING)
            ],
            "QAPairs": [
                ("document_id", pymongo.ASCENDING),
                ("folder_id", pymongo.ASCENDING),
                ([("question", "text"), ("answer", "text")]),  # 텍스트 검색용
                ("question_type", pymongo.ASCENDING),
                ("difficulty", pymongo.ASCENDING)
            ],
            "Folder": [
                ("folder_type", pymongo.ASCENDING),
                ("parent_folder_id", pymongo.ASCENDING),
                ("created_at", pymongo.DESCENDING),
                ("last_accessed_at", pymongo.DESCENDING)
            ]
        } 