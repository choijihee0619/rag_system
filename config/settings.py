import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/rag_system")
        self.vector_db_type = os.getenv("VECTOR_DB_TYPE", "faiss")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

settings = Settings()
