from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
from typing import List, Dict, Any
from config.settings import settings
import os

class VectorStore:
    def __init__(self, store_type: str = None):
        self.store_type = store_type or settings.vector_db_type
        self.vectorstore = None
        
    def create_vectorstore(self, documents: List[Document], embeddings):
        """벡터 스토어 생성"""
        if self.store_type == "faiss":
            self.vectorstore = FAISS.from_documents(documents, embeddings)
        elif self.store_type == "chroma":
            self.vectorstore = Chroma.from_documents(
                documents, 
                embeddings,
                persist_directory="./data/embeddings/chroma"
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
        
        return self.vectorstore
    
    def save(self, path: str):
        """벡터 스토어 저장"""
        if self.store_type == "faiss":
            self.vectorstore.save_local(path)
        elif self.store_type == "chroma":
            self.vectorstore.persist()
    
    def load(self, path: str, embeddings):
        """벡터 스토어 로드"""
        if self.store_type == "faiss":
            self.vectorstore = FAISS.load_local(path, embeddings)
        elif self.store_type == "chroma":
            self.vectorstore = Chroma(
                persist_directory=path,
                embedding_function=embeddings
            )
        
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """유사도 검색"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5):
        """스코어와 함께 유사도 검색"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
