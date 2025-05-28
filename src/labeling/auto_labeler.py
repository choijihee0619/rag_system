from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
from langchain.schema import Document
from config.settings import settings
import json

class AutoLabeler:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=0.1
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["text"],
            template="""다음 텍스트를 읽고 주요 주제와 태그를 추출하세요.
            
텍스트: {text}

다음 형식으로 JSON 응답을 제공하세요:
{{
    "main_topic": "주요 주제",
    "tags": ["태그1", "태그2", "태그3"],
    "category": "카테고리"
}}

JSON 응답:"""
        )
    
    def label_document(self, document: Document) -> Dict[str, Any]:
        """단일 문서 라벨링"""
        prompt = self.prompt_template.format(text=document.page_content[:1000])
        response = self.llm.predict(prompt)
        
        try:
            labels = json.loads(response)
            return {
                "chunk_id": document.metadata.get("chunk_id", ""),
                "labels": labels
            }
        except json.JSONDecodeError:
            return {
                "chunk_id": document.metadata.get("chunk_id", ""),
                "labels": {
                    "main_topic": "unknown",
                    "tags": [],
                    "category": "uncategorized"
                }
            }
    
    def label_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """문서 리스트 라벨링"""
        labeled_docs = []
        
        for i, doc in enumerate(documents):
            doc.metadata["chunk_id"] = f"chunk_{i}"
            labels = self.label_document(doc)
            labeled_docs.append(labels)
        
        return labeled_docs
