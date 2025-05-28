from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain.schema import Document
from config.settings import settings
import numpy as np

class Embedder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model
        )
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """문서 리스트를 임베딩 벡터로 변환"""
        texts = [doc.page_content for doc in documents]
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """쿼리를 임베딩 벡터로 변환"""
        return self.embeddings.embed_query(query)
    
    def batch_embed_documents(self, documents: List[Document], batch_size: int = 100):
        """배치 처리로 문서 임베딩"""
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self.embed_documents(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
