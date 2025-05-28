from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
from langchain.schema import Document
from config.settings import settings
import json

class QAGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=0.7
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["text"],
            template="""다음 텍스트를 읽고 3개의 질문과 답변을 생성하세요.

텍스트: {text}

다음 형식으로 JSON 응답을 제공하세요:
{{
    "qa_pairs": [
        {{
            "question": "질문1",
            "answer": "답변1"
        }},
        {{
            "question": "질문2",
            "answer": "답변2"
        }},
        {{
            "question": "질문3",
            "answer": "답변3"
        }}
    ]
}}

JSON 응답:"""
        )
    
    def generate_qa(self, document: Document) -> Dict[str, Any]:
        """단일 문서에서 QA 생성"""
        prompt = self.prompt_template.format(text=document.page_content[:1000])
        response = self.llm.predict(prompt)
        
        try:
            qa_data = json.loads(response)
            return {
                "chunk_id": document.metadata.get("chunk_id", ""),
                "qa_pairs": qa_data.get("qa_pairs", [])
            }
        except json.JSONDecodeError:
            return {
                "chunk_id": document.metadata.get("chunk_id", ""),
                "qa_pairs": []
            }
    
    def generate_qa_batch(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """문서 리스트에서 QA 생성"""
        qa_data_list = []
        
        for doc in documents:
            qa_data = self.generate_qa(doc)
            qa_data_list.append(qa_data)
        
        return qa_data_list
