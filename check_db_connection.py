#!/usr/bin/env python3
import sys
sys.path.append('.')

from pymongo import MongoClient
from config.settings import settings
import traceback

def check_mongodb_detailed():
    """MongoDB 연결 정보 및 구조 상세 확인"""
    try:
        print(f"=== MongoDB 연결 정보 ===")
        print(f"MongoDB URI: {settings.mongodb_uri}")
        
        # 직접 MongoClient로 연결
        client = MongoClient(settings.mongodb_uri)
        
        # 서버 정보 확인
        server_info = client.server_info()
        print(f"MongoDB 서버 버전: {server_info['version']}")
        
        # 데이터베이스 목록 확인
        db_list = client.list_database_names()
        print(f"사용 가능한 데이터베이스: {db_list}")
        
        # 우리가 사용하는 데이터베이스 확인
        db = client.rag_system
        collection_list = db.list_collection_names()
        print(f"rag_system 데이터베이스의 컬렉션들: {collection_list}")
        
        print(f"\n=== 각 컬렉션별 상세 정보 ===")
        
        # chunks 컬렉션
        if 'chunks' in collection_list:
            chunks_coll = db.chunks
            chunks_count = chunks_coll.count_documents({})
            print(f"chunks 컬렉션: {chunks_count}개 문서")
            
            if chunks_count > 0:
                sample = chunks_coll.find_one()
                print(f"chunks 샘플 구조: {list(sample.keys()) if sample else 'None'}")
        
        # labels 컬렉션
        if 'labels' in collection_list:
            labels_coll = db.labels
            labels_count = labels_coll.count_documents({})
            print(f"labels 컬렉션: {labels_count}개 문서")
            
            if labels_count > 0:
                sample = labels_coll.find_one()
                print(f"labels 샘플 구조: {list(sample.keys()) if sample else 'None'}")
        
        # qa_pairs 컬렉션
        if 'qa_pairs' in collection_list:
            qa_coll = db.qa_pairs
            qa_count = qa_coll.count_documents({})
            print(f"qa_pairs 컬렉션: {qa_count}개 문서")
            
            if qa_count > 0:
                sample = qa_coll.find_one()
                print(f"qa_pairs 샘플 구조: {list(sample.keys()) if sample else 'None'}")
        
        print(f"\n=== MongoDB Compass 연결 정보 ===")
        print(f"MongoDB Compass에서 다음 URI로 연결하세요:")
        print(f"{settings.mongodb_uri}")
        print(f"데이터베이스 이름: rag_system")
        print(f"주요 컬렉션: chunks, labels, qa_pairs")
        
        # 인덱스 정보도 확인
        print(f"\n=== 인덱스 정보 ===")
        for coll_name in ['chunks', 'labels', 'qa_pairs']:
            if coll_name in collection_list:
                coll = db[coll_name]
                indexes = list(coll.list_indexes())
                print(f"{coll_name} 컬렉션 인덱스: {[idx['name'] for idx in indexes]}")
        
        client.close()
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print(f"상세 오류:")
        traceback.print_exc()

if __name__ == "__main__":
    check_mongodb_detailed() 