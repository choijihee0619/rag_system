#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.utils.database import MongoDBClient
from config.settings import settings
import traceback

def check_mongodb():
    """MongoDB 연결 및 데이터 확인"""
    try:
        print(f"MongoDB URI: {settings.mongodb_uri}")
        print("MongoDB 연결 중...")
        
        client = MongoDBClient(settings.mongodb_uri)
        
        # 데이터베이스 연결 테스트
        print("연결 성공!")
        
        # 각 컬렉션의 데이터 개수 확인
        chunks_count = client.chunks_collection.count_documents({})
        labels_count = client.labels_collection.count_documents({})
        qa_count = client.qa_collection.count_documents({})
        
        print(f"\n=== 데이터베이스 상태 ===")
        print(f"청크 데이터: {chunks_count}개")
        print(f"라벨 데이터: {labels_count}개")  
        print(f"QA 데이터: {qa_count}개")
        
        # 청크 데이터 샘플 확인
        if chunks_count > 0:
            print(f"\n=== 청크 데이터 샘플 ===")
            sample_chunks = list(client.chunks_collection.find().limit(3))
            for i, chunk in enumerate(sample_chunks):
                print(f"청크 {i+1}:")
                print(f"  - ID: {chunk.get('_id')}")
                print(f"  - 파일명: {chunk.get('filename', 'Unknown')}")
                print(f"  - 텍스트 길이: {len(chunk.get('text', ''))}")
                print(f"  - 생성일: {chunk.get('created_at')}")
                print()
        
        # 라벨 데이터 샘플 확인
        if labels_count > 0:
            print(f"=== 라벨 데이터 샘플 ===")
            sample_labels = list(client.labels_collection.find().limit(2))
            for i, label in enumerate(sample_labels):
                print(f"라벨 {i+1}: {label}")
                print()
        
        # QA 데이터 샘플 확인  
        if qa_count > 0:
            print(f"=== QA 데이터 샘플 ===")
            sample_qa = list(client.qa_collection.find().limit(2))
            for i, qa in enumerate(sample_qa):
                print(f"QA {i+1}: {qa}")
                print()
        
        # 파일별 통계
        print(f"=== 파일별 통계 ===")
        pipeline = [
            {"$group": {"_id": "$filename", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        file_stats = list(client.chunks_collection.aggregate(pipeline))
        for stat in file_stats:
            print(f"파일: {stat['_id']} -> {stat['count']}개 청크")
        
        client.close()
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print(f"상세 오류:")
        traceback.print_exc()

if __name__ == "__main__":
    check_mongodb() 