#!/usr/bin/env python3
"""
간단한 정리 스크립트
CREATED [2024-12-19]: 마이그레이션 완료 후 기존 컬렉션과 백업 파일 정리
"""

import sys
import os
import shutil
import glob
sys.path.append('.')

from pymongo import MongoClient
from config.settings import settings
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCleanup:
    """간단한 정리 클래스"""
    
    def __init__(self, mongodb_uri: str):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.rag_system
        
        # 삭제 대상 컬렉션
        self.old_collections = ['chunks', 'labels', 'qa_pairs']
        
        # 백업 파일 경로
        self.backup_paths = [
            './data/backup',
            './data/migration'
        ]
    
    def show_current_status(self):
        """현재 상태 표시"""
        logger.info("📊 현재 데이터베이스 상태:")
        
        all_collections = self.db.list_collection_names()
        
        # 기존 컬렉션 (삭제 예정)
        print("\n🗑️ 삭제 예정 컬렉션:")
        total_old_docs = 0
        for col_name in self.old_collections:
            if col_name in all_collections:
                count = self.db[col_name].count_documents({})
                total_old_docs += count
                print(f"  ❌ {col_name}: {count:,}개 문서")
            else:
                print(f"  ⏭️ {col_name}: 이미 없음")
        
        # 새 컬렉션 (유지)
        print("\n✅ 유지할 컬렉션:")
        new_collections = ['Document', 'Labels', 'QAPairs', 'Folder']
        total_new_docs = 0
        for col_name in new_collections:
            if col_name in all_collections:
                count = self.db[col_name].count_documents({})
                total_new_docs += count
                print(f"  ✅ {col_name}: {count:,}개 문서")
            else:
                print(f"  ❌ {col_name}: 없음 (문제!)")
        
        print(f"\n📈 요약:")
        print(f"  - 삭제될 문서: {total_old_docs:,}개")
        print(f"  - 유지될 문서: {total_new_docs:,}개")
        
        return total_old_docs, total_new_docs
    
    def show_backup_files(self):
        """백업 파일 상태 표시"""
        print("\n🗂️ 백업 파일 상태:")
        
        total_files = 0
        total_size = 0
        
        for backup_path in self.backup_paths:
            if os.path.exists(backup_path):
                # JSON 백업 파일들
                json_files = glob.glob(os.path.join(backup_path, "*.json"))
                for file_path in json_files:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    total_files += 1
                    size_mb = file_size / (1024 * 1024)
                    print(f"  📄 {os.path.basename(file_path)}: {size_mb:.1f}MB")
        
        if total_files == 0:
            print("  📭 백업 파일 없음")
        else:
            total_size_mb = total_size / (1024 * 1024)
            print(f"\n📊 백업 파일 총계: {total_files}개 파일, {total_size_mb:.1f}MB")
        
        return total_files, total_size
    
    def delete_old_collections(self):
        """기존 컬렉션 삭제"""
        logger.info("🗑️ 기존 컬렉션 삭제 중...")
        
        deleted = []
        for col_name in self.old_collections:
            if col_name in self.db.list_collection_names():
                count = self.db[col_name].count_documents({})
                self.db[col_name].drop()
                deleted.append((col_name, count))
                logger.info(f"✅ {col_name} 삭제 완료 ({count:,}개 문서)")
        
        return deleted
    
    def delete_backup_files(self):
        """백업 파일 삭제"""
        logger.info("🗂️ 백업 파일 삭제 중...")
        
        deleted_files = 0
        for backup_path in self.backup_paths:
            if os.path.exists(backup_path):
                try:
                    # 디렉토리 전체 삭제
                    shutil.rmtree(backup_path)
                    logger.info(f"✅ {backup_path} 디렉토리 삭제 완료")
                    deleted_files += 1
                except Exception as e:
                    logger.warning(f"⚠️ {backup_path} 삭제 실패: {str(e)}")
        
        return deleted_files
    
    def quick_cleanup(self):
        """빠른 정리 실행"""
        logger.info("🚀 빠른 정리 시작")
        logger.info("=" * 50)
        
        try:
            # 1. 현재 상태 확인
            old_docs, new_docs = self.show_current_status()
            file_count, file_size = self.show_backup_files()
            
            # 2. 안전성 확인
            if new_docs == 0:
                logger.error("❌ 새 컬렉션에 데이터가 없습니다! 정리를 중단합니다.")
                return False
            
            # 3. 사용자 확인
            print("\n" + "=" * 50)
            print("⚠️  다음 항목들을 삭제하시겠습니까?")
            print(f"   - 기존 컬렉션 3개 ({old_docs:,}개 문서)")
            if file_count > 0:
                print(f"   - 백업 파일 {file_count}개 ({file_size/(1024*1024):.1f}MB)")
            print("=" * 50)
            
            user_input = input("삭제하려면 'YES'를 입력하세요: ")
            
            if user_input.upper() != "YES":
                logger.info("❌ 사용자가 삭제를 취소했습니다.")
                return False
            
            # 4. 기존 컬렉션 삭제
            deleted_collections = self.delete_old_collections()
            
            # 5. 백업 파일 삭제
            deleted_backup_dirs = self.delete_backup_files()
            
            # 6. 결과 요약
            logger.info("=" * 50)
            logger.info("🎉 정리 완료!")
            logger.info(f"   - 삭제된 컬렉션: {len(deleted_collections)}개")
            logger.info(f"   - 삭제된 백업 디렉토리: {deleted_backup_dirs}개")
            logger.info("   - 새 v2 스키마만 남아있습니다!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 정리 중 오류 발생: {str(e)}")
            return False
        finally:
            self.client.close()

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="간단한 정리 스크립트")
    parser.add_argument("--status-only", action="store_true", help="상태 확인만 실행")
    parser.add_argument("--mongodb-uri", default=settings.mongodb_uri, help="MongoDB URI")
    
    args = parser.parse_args()
    
    cleanup = SimpleCleanup(args.mongodb_uri)
    
    if args.status_only:
        cleanup.show_current_status()
        cleanup.show_backup_files()
    else:
        cleanup.quick_cleanup()

if __name__ == "__main__":
    main() 