from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    query: str
    k: int = Field(default=5)
    filters: Optional[Dict[str, Any]] = Field(default=None)

class Source(BaseModel):
    chunk_id: str
    text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    metadata: Dict[str, Any]

class EmbedRequest(BaseModel):
    text: str
    chunk_size: int = Field(default=500)
    overlap: int = Field(default=50)

class EmbedResponse(BaseModel):
    chunks: List[str]
    embedding_dims: int
    num_chunks: int

class DocumentInfo(BaseModel):
    id: str
    title: str
    chunks: int
    created_at: str

class LabelRequest(BaseModel):
    text: str

class LabelResponse(BaseModel):
    main_topic: str
    tags: List[str]
    category: str

class QARequest(BaseModel):
    text: str
    num_pairs: int = Field(default=3)

class QAResponse(BaseModel):
    qa_pairs: List[Dict[str, str]]
