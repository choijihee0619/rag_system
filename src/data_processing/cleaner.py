import re
from typing import List
from langchain.schema import Document

class TextCleaner:
    def __init__(self):
        self.patterns = {
            'ads': r'이 기사에는 광고가 포함될 수 있습니다\.',
            'html': r'<[^>]+>',
            'special_chars': r'[^\w\s가-힣]',
            'multiple_spaces': r'\s+',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        }
    
    def clean_text(self, text: str) -> str:
        """텍스트 클렌징"""
        # 광고 제거
        text = re.sub(self.patterns['ads'], '', text)
        
        # HTML 태그 제거
        text = re.sub(self.patterns['html'], '', text)
        
        # URL 제거
        text = re.sub(self.patterns['urls'], '', text)
        
        # 다중 공백 제거
        text = re.sub(self.patterns['multiple_spaces'], ' ', text)
        
        return text.strip()
    
    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """문서 리스트 클렌징"""
        cleaned_docs = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            if cleaned_text:  # 빈 문자열이 아닌 경우만 포함
                doc.page_content = cleaned_text
                cleaned_docs.append(doc)
        
        return cleaned_docs
