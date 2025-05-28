from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.api.schemas import QueryRequest, QueryResponse
from src.retrieval.rag_engine import RAGEngine
from src.retrieval.retriever import Retriever
from src.embedding.vectorstore import VectorStore
from src.embedding.embedder import Embedder
from src.utils.database import MongoDBClient
from config.settings import settings
import uvicorn
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 전역 객체 초기화
logger.debug("Initializing global objects...")
try:
    embedder = Embedder()
    vectorstore = VectorStore()
    db_client = MongoDBClient(settings.mongodb_uri)
    
    # 벡터 스토어 로드 시도
    try:
        logger.debug("Loading vector store...")
        vectorstore.load("./data/embeddings/faiss", embedder.embeddings)
        logger.debug("Vector store loaded successfully")
    except Exception as e:
        logger.warning(f"Vector store initialization failed, continuing in basic mode: {str(e)}")

    retriever = Retriever(vectorstore, db_client)
    rag_engine = RAGEngine(retriever)
    logger.debug("Global objects initialized successfully")
except Exception as e:
    logger.error(f"Error initializing global objects: {str(e)}")
    # 기본 모드로라도 동작하도록 최소한의 초기화
    try:
        embedder = Embedder()
        vectorstore = None
        db_client = None
        retriever = None
        rag_engine = None
        logger.warning("Running in fallback mode with LLM only")
    except Exception as e2:
        logger.error(f"Complete initialization failure: {str(e2)}")
        rag_engine = None

# FastAPI 앱 생성
logger.debug("Creating FastAPI application...")
app = FastAPI(
    title="RAG System API",
    description="Retrieval Augmented Generation System with LangChain",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS 미들웨어 설정
logger.debug("Adding CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 추가
logger.debug("Adding API router...")
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "RAG System API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_db": settings.vector_db_type,
        "llm_model": settings.llm_model,
        "rag_engine_initialized": rag_engine is not None,
        "mode": "rag" if rag_engine is not None else "llm_only"
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """사용자 질문 처리"""
    try:
        if rag_engine is None:
            # RAG 엔진이 없으면 기본 LLM으로 응답
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model=settings.llm_model,
                temperature=0.7
            )
            answer = llm.predict(request.query)
            
            return QueryResponse(
                answer=answer,
                sources=[],
                metadata={
                    "model": settings.llm_model,
                    "total_sources": 0,
                    "mode": "llm_only"
                }
            )
            
        logger.debug(f"Processing query: {request.query}")
        result = rag_engine.invoke({"query": request.query})
        
        return QueryResponse(
            answer=result["answer"],
            sources=[{
                "chunk_id": f"chunk_{i}",
                "text": doc.page_content[:200] + "..." if hasattr(doc, 'page_content') else str(doc)[:200] + "...",
                "score": 0.95 - (i * 0.05)
            } for i, doc in enumerate(result.get("sources", []))],
            metadata={
                "model": settings.llm_model,
                "total_sources": len(result.get("sources", [])),
                "mode": "rag" if result.get("sources") else "llm_only"
            }
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
