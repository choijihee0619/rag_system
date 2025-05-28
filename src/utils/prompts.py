from langchain.prompts import PromptTemplate

# RAG 프롬프트 템플릿
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""다음 컨텍스트를 참고하여 질문에 답변하세요.
답변은 정확하고 도움이 되도록 작성하세요.
컨텍스트에 답변이 없다면, 모른다고 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
)

# 라벨링 프롬프트 템플릿
LABELING_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""다음 텍스트를 분석하여 주제와 태그를 추출하세요.

텍스트: {text}

다음 형식으로 응답하세요:
- 주요 주제: 
- 태그 (콤마로 구분): 
- 카테고리:"""
)

# QA 생성 프롬프트 템플릿
QA_GENERATION_PROMPT = PromptTemplate(
    input_variables=["text", "num_pairs"],
    template="""다음 텍스트를 읽고 {num_pairs}개의 질문과 답변을 생성하세요.
질문은 명확하고 구체적이어야 하며, 답변은 텍스트의 내용을 기반으로 해야 합니다.

텍스트: {text}

질문과 답변을 생성하세요:"""
)

# 의도 분석 프롬프트 템플릿
INTENT_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""다음 사용자 질문의 의도를 분석하세요.

질문: {query}

분석 결과:
- 주요 의도: 
- 정보 유형: 
- 키워드:"""
)
