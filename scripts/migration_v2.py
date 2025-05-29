#!/usr/bin/env python3
"""
MongoDB 데이터베이스 마이그레이션 스크립트 v2.0
CREATED [2024-12-19]: 기존 컬렉션 구조를 새로운 구조로 마이그레이션

마이그레이션 내용:
- chunks → Document (folder_id 추가, 필드명 변경)
- labels → Labels (구조 평탄화, folder_id 추가)
- qa_pairs → QAPairs (배열을 개별 문서로 분리)
- 신규 Folder 컬렉션 생성
"""

import sys
import os
sys.path.append('.')

from pymongo import MongoClient
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import json
import logging
from bson import ObjectId
from config.settings import settings
from src.utils.schemas import MongoSchemas

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseMigration:
    def __init__(self, mongodb_uri: str):
        """마이그레이션 클래스 초기화"""
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.rag_system
        
        # 기존 컬렉션
        self.old_chunks = self.db.chunks
        self.old_labels = self.db.labels
        self.old_qa_pairs = self.db.qa_pairs
        
        # 새 컬렉션
        self.new_documents = self.db.Document
        self.new_labels = self.db.Labels
        self.new_qa_pairs = self.db.QAPairs
        self.new_folders = self.db.Folder
        
        # 기본 폴더 ID (마이그레이션 시 생성)
        self.default_folder_id = None
        
    def backup_existing_data(self, backup_path: str = "./data/backup"):
        """기존 데이터 백업"""
        logger.info("🔄 기존 데이터 백업 시작...")
        
        os.makedirs(backup_path, exist_ok=True)
        
        # 각 컬렉션 백업
        collections = {
            'chunks': self.old_chunks,
            'labels': self.old_labels,
            'qa_pairs': self.old_qa_pairs
        }
        
        for name, collection in collections.items():
            data = list(collection.find())
            backup_file = os.path.join(backup_path, f"{name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            # ObjectId를 문자열로 변환
            for doc in data:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"✅ {name} 백업 완료: {backup_file} ({len(data)}개 문서)")
        
        logger.info("✅ 모든 데이터 백업 완료")
    
    def create_default_folder(self) -> ObjectId:
        """기본 폴더 생성"""
        logger.info("📁 기본 폴더 생성...")
        
        default_folder = {
            "title": "기본 폴더",
            "description": "마이그레이션으로 생성된 기본 컨테이너 폴더",
            "folder_type": "general",
            "parent_folder_id": None,
            "created_at": datetime.now(timezone.utc),
            "last_accessed_at": datetime.now(timezone.utc),
            "cover_image_url": None,
            "metadata": {
                "migration_created": True,
                "migration_date": datetime.now(timezone.utc).isoformat()
            }
        }
        
        result = self.new_folders.insert_one(default_folder)
        self.default_folder_id = result.inserted_id
        
        logger.info(f"✅ 기본 폴더 생성 완료: {self.default_folder_id}")
        return self.default_folder_id
    
    def migrate_chunks_to_documents(self):
        """chunks → Document 마이그레이션"""
        logger.info("🔄 chunks → Document 마이그레이션 시작...")
        
        chunks = list(self.old_chunks.find())
        logger.info(f"📊 처리할 chunks: {len(chunks)}개")
        
        migrated_count = 0
        chunk_id_mapping = {}  # 기존 chunk_id와 새 document ObjectId 매핑
        
        for chunk in chunks:
            try:
                # 새 Document 데이터 구조
                new_document = {
                    "folder_id": self.default_folder_id,
                    "chunk_sequence": chunk.get("chunk_id", f"chunk_{migrated_count}"),
                    "raw_text": chunk.get("content", ""),
                    "text_embedding": [],  # 임시로 빈 배열 (나중에 벡터 이관)
                    "metadata": chunk.get("metadata", {}),
                    "created_at": chunk.get("created_at", datetime.now(timezone.utc)),
                    "updated_at": datetime.now(timezone.utc)
                }
                
                # 메타데이터에 원본 정보 추가
                new_document["metadata"]["original_chunk_id"] = chunk.get("chunk_id")
                new_document["metadata"]["migrated_from"] = "chunks"
                
                # 새 Document 삽입
                result = self.new_documents.insert_one(new_document)
                
                # ID 매핑 저장 (나중에 Labels, QAPairs 마이그레이션에서 사용)
                original_chunk_id = chunk.get("chunk_id")
                if original_chunk_id:
                    chunk_id_mapping[original_chunk_id] = result.inserted_id
                
                migrated_count += 1
                
                if migrated_count % 100 == 0:
                    logger.info(f"진행 상황: {migrated_count}/{len(chunks)} chunks 마이그레이션 완료")
                    
            except Exception as e:
                logger.error(f"❌ chunk 마이그레이션 실패: {chunk.get('_id', 'unknown')} - {str(e)}")
        
        logger.info(f"✅ chunks → Document 마이그레이션 완료: {migrated_count}개")
        return chunk_id_mapping
    
    def migrate_labels(self, chunk_id_mapping: Dict[str, ObjectId]):
        """labels → Labels 마이그레이션"""
        logger.info("🔄 labels → Labels 마이그레이션 시작...")
        
        labels = list(self.old_labels.find())
        logger.info(f"📊 처리할 labels: {len(labels)}개")
        
        migrated_count = 0
        
        for label in labels:
            try:
                # 기존 chunk_id로 새 document_id 찾기
                original_chunk_id = label.get("chunk_id")
                document_id = chunk_id_mapping.get(original_chunk_id)
                
                if not document_id:
                    logger.warning(f"⚠️ chunk_id '{original_chunk_id}'에 해당하는 Document를 찾을 수 없음")
                    continue
                
                # 라벨 데이터 구조 평탄화
                labels_data = label.get("labels", {})
                
                new_label = {
                    "document_id": document_id,
                    "folder_id": self.default_folder_id,
                    "main_topic": labels_data.get("main_topic", ""),
                    "tags": labels_data.get("tags", []),
                    "category": labels_data.get("category", "general"),
                    "confidence": 0.8,  # 기본 신뢰도
                    "created_at": label.get("created_at", datetime.now(timezone.utc)),
                    "updated_at": datetime.now(timezone.utc)
                }
                
                # 새 Labels 삽입
                self.new_labels.insert_one(new_label)
                migrated_count += 1
                
                if migrated_count % 50 == 0:
                    logger.info(f"진행 상황: {migrated_count}/{len(labels)} labels 마이그레이션 완료")
                    
            except Exception as e:
                logger.error(f"❌ label 마이그레이션 실패: {label.get('_id', 'unknown')} - {str(e)}")
        
        logger.info(f"✅ labels → Labels 마이그레이션 완료: {migrated_count}개")
    
    def migrate_qa_pairs(self, chunk_id_mapping: Dict[str, ObjectId]):
        """qa_pairs → QAPairs 마이그레이션 (배열을 개별 문서로 분리)"""
        logger.info("🔄 qa_pairs → QAPairs 마이그레이션 시작...")
        
        qa_records = list(self.old_qa_pairs.find())
        logger.info(f"📊 처리할 qa_pairs 레코드: {len(qa_records)}개")
        
        migrated_count = 0
        total_qa_pairs = 0
        
        for qa_record in qa_records:
            try:
                # 기존 chunk_id로 새 document_id 찾기
                original_chunk_id = qa_record.get("chunk_id")
                document_id = chunk_id_mapping.get(original_chunk_id)
                
                if not document_id:
                    logger.warning(f"⚠️ chunk_id '{original_chunk_id}'에 해당하는 Document를 찾을 수 없음")
                    continue
                
                # qa_pairs 배열에서 각 QA를 개별 문서로 분리
                qa_pairs = qa_record.get("qa_pairs", [])
                
                for qa_pair in qa_pairs:
                    if isinstance(qa_pair, dict) and "question" in qa_pair and "answer" in qa_pair:
                        new_qa = {
                            "document_id": document_id,
                            "folder_id": self.default_folder_id,
                            "question": qa_pair["question"],
                            "answer": qa_pair["answer"],
                            "question_type": self._classify_question_type(qa_pair["question"]),
                            "difficulty": "medium",  # 기본 난이도
                            "created_at": qa_record.get("created_at", datetime.now(timezone.utc)),
                            "updated_at": datetime.now(timezone.utc)
                        }
                        
                        # 새 QAPairs 삽입
                        self.new_qa_pairs.insert_one(new_qa)
                        total_qa_pairs += 1
                
                migrated_count += 1
                
                if migrated_count % 10 == 0:
                    logger.info(f"진행 상황: {migrated_count}/{len(qa_records)} qa_records 마이그레이션 완료")
                    
            except Exception as e:
                logger.error(f"❌ qa_pairs 마이그레이션 실패: {qa_record.get('_id', 'unknown')} - {str(e)}")
        
        logger.info(f"✅ qa_pairs → QAPairs 마이그레이션 완료: {total_qa_pairs}개 QA 쌍")
    
    def _classify_question_type(self, question: str) -> str:
        """질문 유형 분류"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["what", "무엇", "뭐"]):
            return "what"
        elif any(word in question_lower for word in ["how", "어떻게", "방법"]):
            return "how"
        elif any(word in question_lower for word in ["why", "왜", "이유"]):
            return "why"
        elif any(word in question_lower for word in ["when", "언제", "시기"]):
            return "when"
        elif any(word in question_lower for word in ["where", "어디", "장소"]):
            return "where"
        elif any(word in question_lower for word in ["who", "누구", "누가"]):
            return "who"
        else:
            return "general"
    
    def create_indexes(self):
        """새 컬렉션에 인덱스 생성"""
        logger.info("🔍 인덱스 생성 시작...")
        
        schemas = MongoSchemas()
        indexes = schemas.get_indexes()
        
        for collection_name, index_list in indexes.items():
            collection = getattr(self, f"new_{collection_name.lower()}", None)
            if collection is None:
                if collection_name == "Document":
                    collection = self.new_documents
                elif collection_name == "Labels":
                    collection = self.new_labels
                elif collection_name == "QAPairs":
                    collection = self.new_qa_pairs
                elif collection_name == "Folder":
                    collection = self.new_folders
            
            if collection is not None:
                for index_spec in index_list:
                    try:
                        if isinstance(index_spec, tuple) and len(index_spec) == 2:
                            collection.create_index(index_spec)
                        elif isinstance(index_spec, list):
                            collection.create_index(index_spec)
                        logger.info(f"✅ {collection_name} 인덱스 생성: {index_spec}")
                    except Exception as e:
                        logger.warning(f"⚠️ {collection_name} 인덱스 생성 실패: {index_spec} - {str(e)}")
        
        logger.info("✅ 인덱스 생성 완료")
    
    def verify_migration(self) -> Dict[str, Any]:
        """마이그레이션 결과 검증"""
        logger.info("🔍 마이그레이션 결과 검증 시작...")
        
        verification_result = {
            "original_counts": {
                "chunks": self.old_chunks.count_documents({}),
                "labels": self.old_labels.count_documents({}),
                "qa_pairs": self.old_qa_pairs.count_documents({})
            },
            "new_counts": {
                "Document": self.new_documents.count_documents({}),
                "Labels": self.new_labels.count_documents({}),
                "QAPairs": self.new_qa_pairs.count_documents({}),
                "Folder": self.new_folders.count_documents({})
            },
            "default_folder_id": str(self.default_folder_id),
            "migration_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # 검증 결과 출력
        logger.info("📊 마이그레이션 결과:")
        logger.info(f"   - 기존 chunks: {verification_result['original_counts']['chunks']}개")
        logger.info(f"   - 새 Document: {verification_result['new_counts']['Document']}개")
        logger.info(f"   - 기존 labels: {verification_result['original_counts']['labels']}개")
        logger.info(f"   - 새 Labels: {verification_result['new_counts']['Labels']}개")
        logger.info(f"   - 기존 qa_pairs: {verification_result['original_counts']['qa_pairs']}개")
        logger.info(f"   - 새 QAPairs: {verification_result['new_counts']['QAPairs']}개")
        logger.info(f"   - 새 Folder: {verification_result['new_counts']['Folder']}개")
        
        return verification_result
    
    def run_migration(self, backup: bool = True):
        """전체 마이그레이션 실행"""
        logger.info("🚀 MongoDB 마이그레이션 시작")
        logger.info("=" * 50)
        
        try:
            # 1. 백업
            if backup:
                self.backup_existing_data()
            
            # 2. 기본 폴더 생성
            self.create_default_folder()
            
            # 3. chunks → Document 마이그레이션
            chunk_id_mapping = self.migrate_chunks_to_documents()
            
            # 4. labels → Labels 마이그레이션
            self.migrate_labels(chunk_id_mapping)
            
            # 5. qa_pairs → QAPairs 마이그레이션
            self.migrate_qa_pairs(chunk_id_mapping)
            
            # 6. 인덱스 생성
            self.create_indexes()
            
            # 7. 검증
            verification_result = self.verify_migration()
            
            logger.info("=" * 50)
            logger.info("🎉 마이그레이션 성공적으로 완료!")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ 마이그레이션 실패: {str(e)}")
            raise
        finally:
            self.client.close()

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MongoDB 데이터베이스 마이그레이션")
    parser.add_argument("--no-backup", action="store_true", help="백업 생성 건너뛰기")
    parser.add_argument("--mongodb-uri", default=settings.mongodb_uri, help="MongoDB URI")
    
    args = parser.parse_args()
    
    # 마이그레이션 실행
    migration = DatabaseMigration(args.mongodb_uri)
    result = migration.run_migration(backup=not args.no_backup)
    
    # 결과 저장
    with open("./data/migration_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n📄 마이그레이션 결과가 ./data/migration_result.json에 저장되었습니다.")

if __name__ == "__main__":
    main() 