#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.utils.database import MongoDBClient
from config.settings import settings

def setup_database():
    """데이터베이스 초기 설정"""
    print("Setting up MongoDB...")
    
    client = MongoDBClient(settings.mongodb_uri)
    
    # 인덱스 생성
    client.chunks_collection.create_index("chunk_id")
    client.labels_collection.create_index("labels.tags")
    client.qa_collection.create_index([("qa_pairs.question", "text")])
    
    print("Indexes created successfully!")
    
    # 샘플 데이터 삽입 (선택사항)
    sample_chunk = {
        "chunk_id": "sample_001",
        "content": "이것은 샘플 청크입니다.",
        "metadata": {"source": "sample.txt"}
    }
    
    client.insert_chunk(sample_chunk)
    print("Sample data inserted!")
    
    client.close()
    print("Database setup completed!")

if __name__ == "__main__":
    setup_database()
