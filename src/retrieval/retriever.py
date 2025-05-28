from typing import List, Dict, Any
from langchain.schema import Document
from src.embedding.vectorstore import VectorStore
from src.utils.database import MongoDBClient

class Retriever:
    def __init__(self, vectorstore: VectorStore, db_client: MongoDBClient):
        self.vectorstore = vectorstore
        self.db_client = db_client
    
    def retrieve_by_similarity(self, query: str, k: int = 5) -> List[Document]:
        """벡터 유사도 기반 검색"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def retrieve_by_labels(self, labels: List[str]) -> List[Dict[str, Any]]:
        """라벨 기반 검색"""
        return self.db_client.find_by_labels(labels)
    
    def retrieve_qa_pairs(self, query: str) -> List[Dict[str, Any]]:
        """QA 데이터베이스에서 검색"""
        return self.db_client.search_qa(query)
    
    def hybrid_retrieve(self, query: str, k: int = 5) -> Dict[str, Any]:
        """하이브리드 검색 (벡터 + 라벨 + QA)"""
        # 벡터 유사도 검색
        similar_docs = self.retrieve_by_similarity(query, k=k)
        
        # QA 검색
        qa_results = self.retrieve_qa_pairs(query)
        
        # 결과 조합
        return {
            "similar_documents": similar_docs,
            "qa_pairs": qa_results,
            "total_results": len(similar_docs) + len(qa_results)
        }
