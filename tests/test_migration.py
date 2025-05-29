#!/usr/bin/env python3
"""
마이그레이션 테스트 스크립트
CREATED [2024-12-19]: 마이그레이션 결과 검증 및 기능 테스트
"""

import sys
import os
sys.path.append('.')

from src.utils.database_v2 import MongoDBClientV2
from src.utils.database import MongoDBClient
from src.embedding.vectorstore_v2 import MongoVectorStore
from config.settings import settings
import logging
from bson import ObjectId

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MigrationTest:
    """마이그레이션 테스트 클래스"""
    
    def __init__(self):
        self.db_v2 = MongoDBClientV2(settings.mongodb_uri)
        self.db_legacy = MongoDBClient(settings.mongodb_uri)
        self.vector_store = MongoVectorStore(self.db_v2)
    
    def test_data_counts(self):
        """데이터 개수 검증"""
        logger.info("📊 데이터 개수 검증 시작...")
        
        # 새 컬렉션 개수
        document_count = self.db_v2.documents.count_documents({})
        labels_count = self.db_v2.labels.count_documents({})
        qa_pairs_count = self.db_v2.qa_pairs.count_documents({})
        folder_count = self.db_v2.folders.count_documents({})
        
        logger.info(f"✅ 새 컬렉션 개수:")
        logger.info(f"   - Document: {document_count}개")
        logger.info(f"   - Labels: {labels_count}개") 
        logger.info(f"   - QAPairs: {qa_pairs_count}개")
        logger.info(f"   - Folder: {folder_count}개")
        
        # 기존 컬렉션과 비교
        old_chunks = self.db_v2.client.rag_system.chunks.count_documents({})
        old_labels = self.db_v2.client.rag_system.labels.count_documents({})
        old_qa_pairs = self.db_v2.client.rag_system.qa_pairs.count_documents({})
        
        logger.info(f"📋 기존 컬렉션 개수:")
        logger.info(f"   - chunks: {old_chunks}개")
        logger.info(f"   - labels: {old_labels}개")
        logger.info(f"   - qa_pairs: {old_qa_pairs}개")
        
        return {
            "new": {"documents": document_count, "labels": labels_count, "qa_pairs": qa_pairs_count, "folders": folder_count},
            "old": {"chunks": old_chunks, "labels": old_labels, "qa_pairs": old_qa_pairs}
        }
    
    def test_folder_functionality(self):
        """폴더 기능 테스트"""
        logger.info("📁 폴더 기능 테스트 시작...")
        
        # 테스트 폴더 생성
        test_folder_id = self.db_v2.create_folder(
            title="테스트 폴더",
            folder_type="test",
            description="마이그레이션 테스트용 폴더"
        )
        
        # 폴더 조회
        folder = self.db_v2.get_folder(test_folder_id)
        assert folder is not None, "폴더 조회 실패"
        assert folder["title"] == "테스트 폴더", "폴더 제목 불일치"
        
        # 폴더 목록 조회
        folders = self.db_v2.list_folders()
        assert len(folders) > 0, "폴더 목록 조회 실패"
        
        logger.info("✅ 폴더 기능 테스트 통과")
        return test_folder_id
    
    def test_document_operations(self, folder_id: ObjectId):
        """문서 작업 테스트"""
        logger.info("📄 문서 작업 테스트 시작...")
        
        # 테스트 문서 삽입
        doc_id = self.db_v2.insert_document(
            folder_id=folder_id,
            raw_text="이것은 테스트 문서입니다.",
            chunk_sequence="test_chunk_1",
            text_embedding=[0.1, 0.2, 0.3],  # 테스트용 짧은 임베딩
            metadata={"test": True, "source": "migration_test"}
        )
        
        # 문서 조회
        document = self.db_v2.get_document(doc_id)
        assert document is not None, "문서 조회 실패"
        assert document["raw_text"] == "이것은 테스트 문서입니다.", "문서 내용 불일치"
        
        # 폴더별 문서 조회
        folder_docs = self.db_v2.get_documents_by_folder(folder_id)
        assert len(folder_docs) > 0, "폴더별 문서 조회 실패"
        
        logger.info("✅ 문서 작업 테스트 통과")
        return doc_id
    
    def test_labels_operations(self, document_id: ObjectId, folder_id: ObjectId):
        """라벨 작업 테스트"""
        logger.info("🏷️ 라벨 작업 테스트 시작...")
        
        # 라벨 삽입
        label_id = self.db_v2.insert_labels(
            document_id=document_id,
            folder_id=folder_id,
            main_topic="테스트",
            tags=["마이그레이션", "테스트", "MongoDB"],
            category="test"
        )
        
        # 라벨 조회
        labels = self.db_v2.get_labels_by_document(document_id)
        assert len(labels) > 0, "라벨 조회 실패"
        assert labels[0]["main_topic"] == "테스트", "라벨 토픽 불일치"
        
        # 태그 검색
        tag_results = self.db_v2.search_by_tags(["테스트"])
        assert len(tag_results) > 0, "태그 검색 실패"
        
        logger.info("✅ 라벨 작업 테스트 통과")
        return label_id
    
    def test_qa_operations(self, document_id: ObjectId, folder_id: ObjectId):
        """QA 작업 테스트"""
        logger.info("❓ QA 작업 테스트 시작...")
        
        # QA 삽입
        qa_id = self.db_v2.insert_qa_pair(
            document_id=document_id,
            folder_id=folder_id,
            question="이 문서는 무엇에 관한 것인가요?",
            answer="마이그레이션 테스트에 관한 것입니다.",
            question_type="what",
            difficulty="easy"
        )
        
        # QA 조회
        qa_pairs = self.db_v2.get_qa_pairs_by_document(document_id)
        assert len(qa_pairs) > 0, "QA 조회 실패"
        assert "마이그레이션" in qa_pairs[0]["answer"], "QA 답변 불일치"
        
        # QA 검색
        search_results = self.db_v2.search_qa_pairs("마이그레이션")
        assert len(search_results) > 0, "QA 검색 실패"
        
        logger.info("✅ QA 작업 테스트 통과")
        return qa_id
    
    def test_hybrid_search(self, folder_id: ObjectId):
        """하이브리드 검색 테스트"""
        logger.info("🔍 하이브리드 검색 테스트 시작...")
        
        # 하이브리드 검색 실행
        results = self.db_v2.hybrid_search("테스트", folder_id, k=3)
        
        assert "text_results" in results, "텍스트 검색 결과 누락"
        assert "qa_results" in results, "QA 검색 결과 누락"
        assert "tag_results" in results, "태그 검색 결과 누락"
        
        logger.info(f"✅ 하이브리드 검색 테스트 통과 (총 {results['total_results']}개 결과)")
        return results
    
    def test_legacy_compatibility(self):
        """기존 코드 호환성 테스트"""
        logger.info("🔄 호환성 테스트 시작...")
        
        # 기존 방식으로 chunks 조회
        chunks = self.db_legacy.get_all_chunks()
        assert len(chunks) > 0, "기존 chunks 조회 실패"
        
        # 기존 방식으로 라벨 검색
        label_results = self.db_legacy.find_by_labels(["테스트"])
        # 결과가 있을 수도 없을 수도 있음 (새로 추가된 테스트 데이터에 따라)
        
        logger.info("✅ 호환성 테스트 통과")
    
    def test_statistics(self, folder_id: ObjectId):
        """통계 기능 테스트"""
        logger.info("📈 통계 기능 테스트 시작...")
        
        # 폴더 통계
        stats = self.db_v2.get_folder_statistics(folder_id)
        assert "document_count" in stats, "문서 개수 통계 누락"
        assert "label_count" in stats, "라벨 개수 통계 누락"
        assert "qa_count" in stats, "QA 개수 통계 누락"
        
        # 전체 폴더 통계
        all_stats = self.db_v2.get_all_folder_stats()
        assert len(all_stats) > 0, "전체 폴더 통계 조회 실패"
        
        logger.info("✅ 통계 기능 테스트 통과")
        return stats
    
    def cleanup_test_data(self, folder_id: ObjectId):
        """테스트 데이터 정리"""
        logger.info("🧹 테스트 데이터 정리 시작...")
        
        # 테스트 폴더와 관련 데이터 삭제
        self.db_v2.delete_folder(folder_id, recursive=True)
        
        logger.info("✅ 테스트 데이터 정리 완료")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("🚀 마이그레이션 테스트 시작")
        logger.info("=" * 50)
        
        try:
            # 1. 데이터 개수 검증
            counts = self.test_data_counts()
            
            # 2. 폴더 기능 테스트
            test_folder_id = self.test_folder_functionality()
            
            # 3. 문서 작업 테스트
            test_doc_id = self.test_document_operations(test_folder_id)
            
            # 4. 라벨 작업 테스트
            test_label_id = self.test_labels_operations(test_doc_id, test_folder_id)
            
            # 5. QA 작업 테스트
            test_qa_id = self.test_qa_operations(test_doc_id, test_folder_id)
            
            # 6. 하이브리드 검색 테스트
            search_results = self.test_hybrid_search(test_folder_id)
            
            # 7. 호환성 테스트
            self.test_legacy_compatibility()
            
            # 8. 통계 기능 테스트
            stats = self.test_statistics(test_folder_id)
            
            # 9. 테스트 데이터 정리
            self.cleanup_test_data(test_folder_id)
            
            logger.info("=" * 50)
            logger.info("🎉 모든 테스트가 성공적으로 완료되었습니다!")
            
            return {
                "status": "success",
                "data_counts": counts,
                "search_results": search_results,
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"❌ 테스트 실패: {str(e)}")
            return {
                "status": "failed", 
                "error": str(e)
            }
        finally:
            self.db_v2.close()
            self.db_legacy.close()

def main():
    """메인 실행 함수"""
    test = MigrationTest()
    result = test.run_all_tests()
    
    if result["status"] == "success":
        print("\n✅ 마이그레이션이 성공적으로 완료되어 모든 기능이 정상 동작합니다!")
    else:
        print(f"\n❌ 테스트 실패: {result['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 