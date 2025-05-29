"""
MongoDB 기반 벡터 스토어 v2.0
CREATED [2024-12-19]: MongoDB text_embedding 필드를 활용한 벡터 검색

기능:
- MongoDB 내장 벡터 검색
- FAISS/Chroma 벡터 DB에서 MongoDB로 임베딩 이관
- 하이브리드 검색 지원
"""

from langchain.schema import Document
from typing import List, Dict, Any, Optional, Union
import numpy as np
from bson import ObjectId
import logging
from src.utils.database_v2 import MongoDBClientV2

logger = logging.getLogger(__name__)

class MongoVectorStore:
    """MongoDB 기반 벡터 스토어"""
    
    def __init__(self, mongodb_client: MongoDBClientV2):
        self.db_client = mongodb_client
        self.collection = mongodb_client.documents
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]], 
                     folder_id: ObjectId) -> List[ObjectId]:
        """문서와 임베딩을 MongoDB에 저장"""
        document_ids = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            document_id = self.db_client.insert_document(
                folder_id=folder_id,
                raw_text=doc.page_content,
                chunk_sequence=doc.metadata.get("chunk_id", f"chunk_{i}"),
                text_embedding=embedding,
                metadata=doc.metadata
            )
            document_ids.append(document_id)
        
        logger.info(f"✅ {len(documents)}개 문서와 임베딩을 MongoDB에 저장완료")
        return document_ids
    
    def similarity_search(self, query_embedding: List[float], 
                         folder_id: Optional[ObjectId] = None, 
                         k: int = 5) -> List[Document]:
        """벡터 유사도 검색"""
        # MongoDB 벡터 검색 사용
        results = self.db_client.vector_search(query_embedding, folder_id, k)
        
        # Document 객체로 변환
        documents = []
        for result in results:
            doc_data = result["document"]
            document = Document(
                page_content=doc_data.get("raw_text", ""),
                metadata={
                    **doc_data.get("metadata", {}),
                    "document_id": str(doc_data["_id"]),
                    "folder_id": str(doc_data["folder_id"]),
                    "similarity": result["similarity"],
                    "chunk_sequence": doc_data.get("chunk_sequence", "")
                }
            )
            documents.append(document)
        
        return documents
    
    def similarity_search_with_score(self, query_embedding: List[float], 
                                   folder_id: Optional[ObjectId] = None, 
                                   k: int = 5) -> List[tuple]:
        """점수와 함께 유사도 검색"""
        results = self.db_client.vector_search(query_embedding, folder_id, k)
        
        scored_documents = []
        for result in results:
            doc_data = result["document"]
            document = Document(
                page_content=doc_data.get("raw_text", ""),
                metadata={
                    **doc_data.get("metadata", {}),
                    "document_id": str(doc_data["_id"]),
                    "folder_id": str(doc_data["folder_id"]),
                    "chunk_sequence": doc_data.get("chunk_sequence", "")
                }
            )
            scored_documents.append((document, result["similarity"]))
        
        return scored_documents
    
    def update_embeddings_batch(self, document_embeddings: Dict[ObjectId, List[float]]):
        """배치로 임베딩 업데이트"""
        updated_count = 0
        
        for document_id, embedding in document_embeddings.items():
            try:
                self.db_client.update_document_embedding(document_id, embedding)
                updated_count += 1
            except Exception as e:
                logger.error(f"임베딩 업데이트 실패 {document_id}: {str(e)}")
        
        logger.info(f"✅ {updated_count}개 문서 임베딩 업데이트 완료")
        return updated_count
    
    def migrate_from_faiss(self, faiss_vectorstore, embedder, folder_id: ObjectId):
        """FAISS에서 MongoDB로 벡터 데이터 이관"""
        logger.info("🔄 FAISS → MongoDB 벡터 이관 시작...")
        
        try:
            # FAISS 인덱스에서 모든 벡터와 문서 가져오기
            # 주의: 실제 구현에서는 FAISS 저장 방식에 따라 조정 필요
            if hasattr(faiss_vectorstore, 'docstore') and hasattr(faiss_vectorstore, 'index_to_docstore_id'):
                migrated_count = 0
                
                # FAISS 저장된 문서들 순회
                for idx in range(faiss_vectorstore.index.ntotal):
                    try:
                        # 문서 ID 가져오기
                        doc_id = faiss_vectorstore.index_to_docstore_id.get(idx)
                        if doc_id is None:
                            continue
                        
                        # 문서 가져오기
                        document = faiss_vectorstore.docstore.search(doc_id)
                        if document is None:
                            continue
                        
                        # 벡터 가져오기
                        vector = faiss_vectorstore.index.reconstruct(idx)
                        embedding = vector.tolist()
                        
                        # MongoDB에 저장
                        document_id = self.db_client.insert_document(
                            folder_id=folder_id,
                            raw_text=document.page_content,
                            chunk_sequence=document.metadata.get("chunk_id", f"migrated_{idx}"),
                            text_embedding=embedding,
                            metadata={
                                **document.metadata,
                                "migrated_from": "faiss",
                                "original_index": idx
                            }
                        )
                        
                        migrated_count += 1
                        
                        if migrated_count % 100 == 0:
                            logger.info(f"진행 상황: {migrated_count}개 벡터 이관 완료")
                            
                    except Exception as e:
                        logger.error(f"인덱스 {idx} 이관 실패: {str(e)}")
                        continue
                
                logger.info(f"✅ FAISS → MongoDB 이관 완료: {migrated_count}개 벡터")
                return migrated_count
                
        except Exception as e:
            logger.error(f"❌ FAISS 이관 실패: {str(e)}")
            return 0
    
    def migrate_from_chroma(self, chroma_vectorstore, folder_id: ObjectId):
        """Chroma에서 MongoDB로 벡터 데이터 이관"""
        logger.info("🔄 Chroma → MongoDB 벡터 이관 시작...")
        
        try:
            # Chroma에서 모든 데이터 가져오기
            if hasattr(chroma_vectorstore, '_collection'):
                # Chroma 컬렉션에서 데이터 가져오기
                collection = chroma_vectorstore._collection
                results = collection.get(include=['documents', 'embeddings', 'metadatas'])
                
                migrated_count = 0
                documents = results.get('documents', [])
                embeddings = results.get('embeddings', [])
                metadatas = results.get('metadatas', [])
                
                for i, (doc, embedding, metadata) in enumerate(zip(documents, embeddings, metadatas)):
                    try:
                        document_id = self.db_client.insert_document(
                            folder_id=folder_id,
                            raw_text=doc,
                            chunk_sequence=metadata.get("chunk_id", f"chroma_migrated_{i}"),
                            text_embedding=embedding,
                            metadata={
                                **(metadata or {}),
                                "migrated_from": "chroma",
                                "original_index": i
                            }
                        )
                        
                        migrated_count += 1
                        
                        if migrated_count % 100 == 0:
                            logger.info(f"진행 상황: {migrated_count}개 벡터 이관 완료")
                            
                    except Exception as e:
                        logger.error(f"인덱스 {i} 이관 실패: {str(e)}")
                        continue
                
                logger.info(f"✅ Chroma → MongoDB 이관 완료: {migrated_count}개 벡터")
                return migrated_count
                
        except Exception as e:
            logger.error(f"❌ Chroma 이관 실패: {str(e)}")
            return 0
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], 
                          folder_id: Optional[ObjectId] = None) -> List[Document]:
        """메타데이터로 검색"""
        query = {}
        
        if folder_id:
            query["folder_id"] = folder_id
        
        # 메타데이터 필터 추가
        for key, value in metadata_filter.items():
            query[f"metadata.{key}"] = value
        
        results = list(self.collection.find(query))
        
        documents = []
        for doc_data in results:
            document = Document(
                page_content=doc_data.get("raw_text", ""),
                metadata={
                    **doc_data.get("metadata", {}),
                    "document_id": str(doc_data["_id"]),
                    "folder_id": str(doc_data["folder_id"]),
                    "chunk_sequence": doc_data.get("chunk_sequence", "")
                }
            )
            documents.append(document)
        
        return documents
    
    def get_document_count(self, folder_id: Optional[ObjectId] = None) -> int:
        """문서 개수 조회"""
        query = {}
        if folder_id:
            query["folder_id"] = folder_id
        
        return self.collection.count_documents(query)
    
    def delete_documents(self, document_ids: List[ObjectId]) -> int:
        """문서 삭제"""
        result = self.collection.delete_many({"_id": {"$in": document_ids}})
        logger.info(f"✅ {result.deleted_count}개 문서 삭제 완료")
        return result.deleted_count

class HybridVectorStore:
    """하이브리드 벡터 스토어 (MongoDB + 기존 벡터 DB)"""
    
    def __init__(self, mongo_vectorstore: MongoVectorStore, 
                 fallback_vectorstore=None):
        self.mongo_store = mongo_vectorstore
        self.fallback_store = fallback_vectorstore
    
    def similarity_search(self, query: str, embedder, k: int = 5, 
                         folder_id: Optional[ObjectId] = None) -> List[Document]:
        """하이브리드 유사도 검색"""
        # 1. 쿼리 임베딩 생성
        query_embedding = embedder.embed_query(query)
        
        # 2. MongoDB에서 검색
        mongo_results = self.mongo_store.similarity_search(query_embedding, folder_id, k)
        
        # 3. 충분한 결과가 없으면 fallback 사용
        if len(mongo_results) < k and self.fallback_store:
            fallback_results = self.fallback_store.similarity_search(query, k - len(mongo_results))
            mongo_results.extend(fallback_results)
        
        return mongo_results[:k]
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]], 
                     folder_id: ObjectId) -> List[ObjectId]:
        """문서 추가 (MongoDB에만 저장)"""
        return self.mongo_store.add_documents(documents, embeddings, folder_id) 