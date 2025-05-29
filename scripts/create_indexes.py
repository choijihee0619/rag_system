#!/usr/bin/env python3
"""
인덱스 생성 스크립트
CREATED [2024-12-19]: 새 컬렉션에 인덱스 생성
"""

import sys
sys.path.append('.')

from pymongo import MongoClient, TEXT, ASCENDING, DESCENDING
from config.settings import settings
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_indexes():
    """새 컬렉션에 인덱스 생성"""
    client = MongoClient(settings.mongodb_uri)
    db = client.rag_system
    
    # 컬렉션들
    documents = db.Document
    labels = db.Labels
    qa_pairs = db.QAPairs
    folders = db.Folder
    
    logger.info("🔍 인덱스 생성 시작...")
    
    try:
        # Document 컬렉션 인덱스
        logger.info("Document 컬렉션 인덱스 생성...")
        documents.create_index([("folder_id", ASCENDING)])
        documents.create_index([("chunk_sequence", ASCENDING)])
        documents.create_index([("created_at", DESCENDING)])
        documents.create_index([("raw_text", TEXT)])
        logger.info("✅ Document 인덱스 생성 완료")
        
        # Labels 컬렉션 인덱스
        logger.info("Labels 컬렉션 인덱스 생성...")
        labels.create_index([("document_id", ASCENDING)])
        labels.create_index([("folder_id", ASCENDING)])
        labels.create_index([("main_topic", ASCENDING)])
        labels.create_index([("tags", ASCENDING)])
        labels.create_index([("category", ASCENDING)])
        logger.info("✅ Labels 인덱스 생성 완료")
        
        # QAPairs 컬렉션 인덱스
        logger.info("QAPairs 컬렉션 인덱스 생성...")
        qa_pairs.create_index([("document_id", ASCENDING)])
        qa_pairs.create_index([("folder_id", ASCENDING)])
        qa_pairs.create_index([("question", TEXT), ("answer", TEXT)])
        qa_pairs.create_index([("question_type", ASCENDING)])
        qa_pairs.create_index([("difficulty", ASCENDING)])
        logger.info("✅ QAPairs 인덱스 생성 완료")
        
        # Folder 컬렉션 인덱스
        logger.info("Folder 컬렉션 인덱스 생성...")
        folders.create_index([("folder_type", ASCENDING)])
        folders.create_index([("parent_folder_id", ASCENDING)])
        folders.create_index([("created_at", DESCENDING)])
        folders.create_index([("last_accessed_at", DESCENDING)])
        logger.info("✅ Folder 인덱스 생성 완료")
        
        logger.info("🎉 모든 인덱스 생성 완료!")
        
    except Exception as e:
        logger.error(f"❌ 인덱스 생성 실패: {str(e)}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    create_indexes() 