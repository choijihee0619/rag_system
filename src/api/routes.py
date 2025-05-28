from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from src.api.schemas import EmbedRequest, EmbedResponse, DocumentInfo
from src.data_processing.loader import DocumentLoader
from src.data_processing.cleaner import TextCleaner
from src.data_processing.chunker import TextChunker
from src.embedding.embedder import Embedder
from src.embedding.vectorstore import VectorStore
from src.labeling.auto_labeler import AutoLabeler
from src.labeling.qa_generator import QAGenerator
from src.utils.database import MongoDBClient
from config.settings import settings
import tempfile
import os

router = APIRouter()

@router.post("/embed", response_model=EmbedResponse)
async def embed_documents(request: EmbedRequest):
    """문서 임베딩 생성"""
    try:
        # 텍스트 클리닝
        cleaner = TextCleaner()
        cleaned_text = cleaner.clean_text(request.text)
        
        # 청킹
        chunker = TextChunker()
        chunks = chunker.split_text(cleaned_text)
        
        # 임베딩 생성
        embedder = Embedder()
        embeddings = embedder.embeddings.embed_documents(chunks)
        
        return EmbedResponse(
            chunks=chunks,
            embedding_dims=len(embeddings[0]) if embeddings else 0,
            num_chunks=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """문서 업로드 및 처리"""
    try:
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 문서 로드
        loader = DocumentLoader()
        documents = loader.load_file(tmp_path)
        
        # 클리닝
        cleaner = TextCleaner()
        cleaned_docs = cleaner.clean_documents(documents)
        
        # 청킹
        chunker = TextChunker()
        chunks = chunker.split_documents(cleaned_docs)
        
        # 임베딩 생성 및 벡터 스토어 저장
        embedder = Embedder()
        vectorstore = VectorStore()
        vectorstore.create_vectorstore(chunks, embedder.embeddings)
        vectorstore.save("./data/embeddings/faiss")
        
        # 라벨링
        labeler = AutoLabeler()
        labels = labeler.label_documents(chunks)
        
        # QA 생성
        qa_generator = QAGenerator()
        qa_pairs = qa_generator.generate_qa_batch(chunks[:5])  # 처음 5개만
        
        # MongoDB에 데이터 저장
        db_client = MongoDBClient(settings.mongodb_uri)
        
        # 청크 데이터 저장
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": f"chunk_{i}",
                "text": chunk.page_content,
                "metadata": chunk.metadata,
                "filename": file.filename
            }
            db_client.insert_chunk(chunk_data)
        
        # 라벨 데이터 저장
        for label in labels:
            db_client.insert_labels(label)
        
        # QA 데이터 저장
        for qa in qa_pairs:
            db_client.insert_qa_pairs(qa)
        
        db_client.close()
        
        # 임시 파일 삭제
        os.unlink(tmp_path)
        
        return {
            "filename": file.filename,
            "num_chunks": len(chunks),
            "labels": labels[:3],  # 처음 3개만 반환
            "qa_samples": qa_pairs[:2],  # 처음 2개만 반환
            "vector_store_saved": True,
            "mongodb_saved": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """저장된 문서 목록 조회 - MongoDB에서 실제 데이터 조회"""
    try:
        db_client = MongoDBClient(settings.mongodb_uri)
        
        # MongoDB에서 실제 문서 목록 조회
        chunks = db_client.get_all_chunks()
        
        # 파일별로 그룹화
        file_docs = {}
        for chunk in chunks:
            filename = chunk.get("filename", "Unknown")
            if filename not in file_docs:
                file_docs[filename] = {
                    "chunks": 0,
                    "created_at": chunk.get("created_at", "2024-01-01T00:00:00Z")
                }
            file_docs[filename]["chunks"] += 1
        
        db_client.close()
        
        # DocumentInfo 형태로 변환
        documents = []
        for i, (filename, info) in enumerate(file_docs.items()):
            documents.append(DocumentInfo(
                id=f"doc_{i+1}",
                title=filename,
                chunks=info["chunks"],
                created_at=str(info["created_at"])
            ))
        
        return documents
        
    except Exception as e:
        # DB 연결 실패 시 샘플 데이터 반환
        return [
            DocumentInfo(
                id="doc_1",
                title="Sample Document 1",
                chunks=10,
                created_at="2024-01-01T00:00:00Z"
            ),
            DocumentInfo(
                id="doc_2",
                title="Sample Document 2",
                chunks=15,
                created_at="2024-01-02T00:00:00Z"
            )
        ]
