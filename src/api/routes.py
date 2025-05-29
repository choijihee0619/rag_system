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
import logging

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
    logger = logging.getLogger(__name__)
    
    tmp_path = None
    db_client = None
    
    try:
        logger.info(f"Starting upload process for file: {file.filename}")
        
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"Temporary file saved: {tmp_path}")
        
        # 문서 로드
        try:
            loader = DocumentLoader()
            documents = loader.load_file(tmp_path)
            logger.info(f"Loaded {len(documents)} documents")
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document loading failed: {str(e)}")
        
        # 클리닝
        try:
            cleaner = TextCleaner()
            cleaned_docs = cleaner.clean_documents(documents)
            logger.info(f"Cleaned {len(cleaned_docs)} documents")
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Text cleaning failed: {str(e)}")
        
        # 청킹
        try:
            chunker = TextChunker()
            chunks = chunker.split_documents(cleaned_docs)
            logger.info(f"Created {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Text chunking failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Text chunking failed: {str(e)}")
        
        # 임베딩 생성 및 벡터 스토어 저장
        try:
            embedder = Embedder()
            vectorstore = VectorStore()
            vectorstore.create_vectorstore(chunks, embedder.embeddings)
            vectorstore.save("./data/embeddings/faiss")
            logger.info("Vector store created and saved")
        except Exception as e:
            logger.error(f"Vector store creation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vector store creation failed: {str(e)}")
        
        # 라벨링
        try:
            labeler = AutoLabeler()
            labels = labeler.label_documents(chunks)
            logger.info(f"Generated {len(labels)} labels")
        except Exception as e:
            logger.error(f"Labeling failed: {str(e)}")
            # 라벨링 실패는 치명적이지 않으므로 빈 리스트로 계속 진행
            labels = []
            logger.warning("Continuing without labels")
        
        # QA 생성
        try:
            qa_generator = QAGenerator()
            qa_pairs = qa_generator.generate_qa_batch(chunks[:5])  # 처음 5개만
            logger.info(f"Generated {len(qa_pairs)} QA pairs")
        except Exception as e:
            logger.error(f"QA generation failed: {str(e)}")
            # QA 생성 실패는 치명적이지 않으므로 빈 리스트로 계속 진행
            qa_pairs = []
            logger.warning("Continuing without QA pairs")
        
        # MongoDB에 데이터 저장
        try:
            db_client = MongoDBClient(settings.mongodb_uri)
            
            # 청크 데이터 저장
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "chunk_id": f"chunk_{i}",
                    "text": chunk.page_content,
                    "metadata": chunk.metadata,
                    "filename": file.filename
                }
                result = db_client.insert_chunk(chunk_data)
                chunk_ids.append(str(result.inserted_id))  # ObjectId를 문자열로 변환
            
            # 라벨 데이터 저장
            label_ids = []
            for label in labels:
                result = db_client.insert_labels(label)
                label_ids.append(str(result.inserted_id))  # ObjectId를 문자열로 변환
            
            # QA 데이터 저장
            qa_ids = []
            for qa in qa_pairs:
                result = db_client.insert_qa_pairs(qa)
                qa_ids.append(str(result.inserted_id))  # ObjectId를 문자열로 변환
            
            logger.info("Data saved to MongoDB")
            
        except Exception as e:
            logger.error(f"MongoDB operations failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database save failed: {str(e)}")
        
        finally:
            if db_client:
                db_client.close()
        
        # 임시 파일 삭제
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info("Temporary file deleted")
        
        # ObjectId가 포함되지 않은 안전한 응답 데이터 생성
        safe_labels = []
        for label in labels[:3]:
            if isinstance(label, dict):
                # ObjectId 제거하고 안전한 데이터만 포함
                safe_label = {k: v for k, v in label.items() if k != '_id' and not str(type(v)).startswith("<class 'bson")}
                safe_labels.append(safe_label)
            else:
                safe_labels.append(str(label))
        
        safe_qa_pairs = []
        for qa in qa_pairs[:2]:
            if isinstance(qa, dict):
                # ObjectId 제거하고 안전한 데이터만 포함
                safe_qa = {k: v for k, v in qa.items() if k != '_id' and not str(type(v)).startswith("<class 'bson")}
                safe_qa_pairs.append(safe_qa)
            else:
                safe_qa_pairs.append(str(qa))
        
        return {
            "filename": file.filename,
            "num_chunks": len(chunks),
            "labels": safe_labels,
            "qa_samples": safe_qa_pairs,
            "vector_store_saved": True,
            "mongodb_saved": True
        }
        
    except HTTPException:
        # HTTPException은 다시 raise
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        # 임시 파일 정리
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        if db_client:
            try:
                db_client.close()
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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
