from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any, List
from config.settings import settings
from src.retrieval.retriever import Retriever

class RAGEngine:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=0.7
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""다음 컨텍스트를 참고하여 질문에 답변하세요.
만약 컨텍스트에 답변이 없다면, 일반적인 지식을 바탕으로 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        )
        
        self.basic_prompt_template = PromptTemplate(
            input_variables=["question"],
            template="""다음 질문에 대해 정확하고 도움이 되는 답변을 제공하세요.

질문: {question}

답변:"""
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 쿼리 처리"""
        query = input.get("query", "")
        if not query:
            raise ValueError("Query is required")
            
        try:
            # 하이브리드 검색 시도
            search_results = self.retriever.hybrid_retrieve(query)
            
            # 컨텍스트 생성
            context_parts = []
            
            # 유사 문서 컨텍스트
            for doc in search_results.get("similar_documents", []):
                context_parts.append(doc.page_content)
            
            # QA 컨텍스트
            for qa in search_results.get("qa_pairs", []):
                context_parts.append(f"Q: {qa['question']}\nA: {qa['answer']}")
            
            if context_parts:
                context = "\n\n".join(context_parts[:5])  # 상위 5개만 사용
                prompt = self.prompt_template.format(context=context, question=query)
                answer = self.llm.predict(prompt)
                
                return {
                    "answer": answer,
                    "sources": search_results.get("similar_documents", [])[:3],
                    "qa_references": search_results.get("qa_pairs", [])[:2],
                    "context_used": context[:500] + "..." if len(context) > 500 else context
                }
            else:
                # 컨텍스트가 없으면 기본 답변 모드
                return self._basic_answer(query)
                
        except Exception as e:
            # 벡터 스토어가 없거나 검색 실패 시 기본 답변 모드
            print(f"Search failed, using basic mode: {str(e)}")
            return self._basic_answer(query)
    
    def _basic_answer(self, query: str) -> Dict[str, Any]:
        """벡터 스토어 없이 기본 LLM 답변 제공"""
        prompt = self.basic_prompt_template.format(question=query)
        answer = self.llm.predict(prompt)
        
        return {
            "answer": answer,
            "sources": [],
            "qa_references": [],
            "context_used": "기본 언어모델 지식 사용"
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """단순한 쿼리 처리 메서드"""
        return self.invoke({"query": query})
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """쿼리 의도 분석"""
        intent_prompt = f"""다음 질문의 의도와 주제를 분석하세요:
질문: {query}

다음 형식으로 답변하세요:
- 주요 의도: 
- 관련 주제:
- 키워드:"""
        
        analysis = self.llm.predict(intent_prompt)
        
        return {
            "query": query,
            "analysis": analysis
        }
