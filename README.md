# RAG System with LangChain and FastAPI

## 아키텍처 요약

본 프로젝트는 GPT-4와 LangChain/FastAPI를 활용한 RAG(Retrieval Augmented Generation) 시스템입니다.

### 주요 구성 요소
- **데이터 처리**: 문서 로딩, 클렌징, 청킹
- **임베딩 생성**: OpenAI Embedding API를 통한 벡터화
- **벡터 저장소**: FAISS/Chroma를 활용한 유사도 검색
- **QA 생성**: GPT-4를 활용한 자동 질문-답변 생성
- **RAG 엔진**: 컨텍스트 기반 답변 생성
- **API 서버**: FastAPI 기반 RESTful API

## 설치 및 설정

### 1. 환경 설정
```bash
# Python 3.9+ 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일 편집하여 API 키 설정
# OPENAI_API_KEY=your-api-key
# MONGODB_URI=mongodb://localhost:27017/rag_system
```

### 3. 데이터베이스 초기화
```bash
python scripts/setup_db.py
```

## 사용법

### 1. 문서 처리 및 임베딩 생성
```bash
# 원본 문서를 data/raw/ 폴더에 배치
python scripts/process_documents.py --input data/raw --output data/processed
```

### 2. API 서버 실행
```bash
python scripts/run_server.py
# 또는
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. API 엔드포인트
- `POST /query`: 사용자 질문 처리
- `POST /embed`: 문서 임베딩 생성
- `GET /health`: 서버 상태 확인

## 실행 예시

### 문서 처리
```python
from src.data_processing.loader import DocumentLoader
from src.data_processing.chunker import TextChunker
from src.embedding.embedder import Embedder

# 문서 로딩
loader = DocumentLoader()
docs = loader.load_directory("data/raw")

# 청킹
chunker = TextChunker(chunk_size=500, overlap=50)
chunks = chunker.split_documents(docs)

# 임베딩 생성
embedder = Embedder()
embeddings = embedder.embed_documents(chunks)
```

### API 호출
```bash
# 질문 요청
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "환불 정책에 대해 알려주세요"}'

# 응답 예시
{
  "answer": "환불은 구매일로부터 7일 이내 가능합니다...",
  "sources": [
    {"chunk_id": "123", "score": 0.95, "text": "..."}
  ],
  "metadata": {
    "processing_time": 1.23,
    "model": "gpt-4"
  }
}
```

### Python 클라이언트 예시
```python
import requests

# RAG 시스템에 질문
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "제품 보증 기간은 얼마나 되나요?"}
)

result = response.json()
print(f"답변: {result['answer']}")
print(f"참조 문서: {len(result['sources'])}개")
```

## 프로젝트 구조 설명

- `src/data_processing/`: 문서 로딩, 클렌징, 청킹 모듈
- `src/embedding/`: 임베딩 생성 및 벡터 저장소 관리
- `src/labeling/`: 자동 라벨링 및 QA 생성
- `src/retrieval/`: RAG 검색 엔진 및 컨텍스트 추출
- `src/api/`: LangServe 기반 API 서버
- `src/utils/`: 데이터베이스 연결, 프롬프트 템플릿 등

## 성능 최적화

- 청크 크기: 500-1000 토큰 권장
- 임베딩 배치 처리: 100개씩 배치 처리
- 벡터 검색: Top-K=5 기본값
- 캐싱: Redis를 통한 응답 캐싱 (선택사항)

## 주의사항

- OpenAI API 비용 관리 필요
- 대용량 문서 처리 시 메모리 사용량 주의
- 벡터 DB 인덱스 정기적 최적화 필요