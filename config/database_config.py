"""
데이터베이스 설정
MODIFIED [2024-12-19]: v2 스키마 지원
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    
    # MongoDB 설정
    mongodb_uri: str = "mongodb://localhost:27017"
    database_name: str = "rag_system"
    
    # 컬렉션 설정 (v2)
    collections: Dict[str, str] = None
    
    # 인덱스 설정
    enable_text_search: bool = True
    enable_vector_search: bool = True
    
    # 벡터 설정
    vector_dimension: int = 1536  # OpenAI embedding 차원
    vector_similarity_threshold: float = 0.7
    
    # 폴더 설정
    default_folder_name: str = "기본 폴더"
    auto_create_folders: bool = True
    
    # 성능 설정
    batch_size: int = 100
    max_connections: int = 10
    connection_timeout: int = 30000  # 30초
    
    # 마이그레이션 설정
    backup_enabled: bool = True
    backup_path: str = "./data/backup"
    migration_log_path: str = "./data/migration"
    
    def __post_init__(self):
        if self.collections is None:
            self.collections = {
                "documents": "Document",
                "labels": "Labels", 
                "qa_pairs": "QAPairs",
                "folders": "Folder",
                
                # 기존 컬렉션 (호환성)
                "chunks": "chunks",
                "old_labels": "labels",
                "old_qa_pairs": "qa_pairs"
            }

# 기본 설정 인스턴스
default_db_config = DatabaseConfig()

# 개발 환경 설정
development_config = DatabaseConfig(
    mongodb_uri="mongodb://localhost:27017",
    database_name="rag_system_dev",
    backup_enabled=True
)

# 프로덕션 환경 설정
production_config = DatabaseConfig(
    mongodb_uri="mongodb://localhost:27017",  # 실제 프로덕션 URI로 변경 필요
    database_name="rag_system",
    max_connections=50,
    batch_size=500,
    backup_enabled=True
)

# 테스트 환경 설정
test_config = DatabaseConfig(
    mongodb_uri="mongodb://localhost:27017",
    database_name="rag_system_test",
    backup_enabled=False,
    auto_create_folders=True
) 