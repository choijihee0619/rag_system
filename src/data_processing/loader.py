from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader as PDFLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from typing import List
from langchain.schema import Document
import os

class DocumentLoader:
    def __init__(self):
        self.loaders = {
            '.txt': TextLoader,
            '.pdf': PDFLoader,
            '.md': UnstructuredFileLoader
        }
    
    def load_directory(self, path: str) -> List[Document]:
        """디렉토리에서 문서 로드"""
        documents = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext in self.loaders:
                    loader = self.loaders[ext](file_path)
                    documents.extend(loader.load())
        
        return documents
    
    def load_file(self, file_path: str) -> List[Document]:
        """단일 파일 로드"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in self.loaders:
            loader = self.loaders[ext](file_path)
            return loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
