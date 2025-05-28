#!/usr/bin/env python3
import sys
sys.path.append('.')

import argparse
from src.data_processing.loader import DocumentLoader
from src.data_processing.cleaner import TextCleaner
from src.data_processing.chunker import TextChunker
from src.embedding.embedder import Embedder
from src.embedding.vectorstore import VectorStore
from src.labeling.auto_labeler import AutoLabeler
from src.labeling.qa_generator import QAGenerator
from src.utils.database import MongoDBClient
from config.settings import settings

def process_documents(input_dir: str, output_dir: str):
    """문서 처리 파이프라인"""
    print(f"Processing documents from {input_dir}...")
    
    # 1. 문서 로드
    loader = DocumentLoader()
    documents = loader.load_directory(input_dir)
    print(f"Loaded {len(documents)} documents")
    
    # 2. 클리닝
    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean_documents(documents)
    print(f"Cleaned {len(cleaned_docs)} documents")
    
    # 3. 청킹
    chunker = TextChunker()
    chunks = chunker.split_documents(cleaned_docs)
    print(f"Created {len(chunks)} chunks")
    
    # 4. 임베딩 생성
    embedder = Embedder()
    print("Creating embeddings...")
    
    # 5. 벡터 스토어 생성
    vectorstore = VectorStore()
    vectorstore.create_vectorstore(chunks, embedder.embeddings)
    vectorstore.save(f"{output_dir}/faiss")
    print("Vector store created and saved")
    
    # 6. 라벨링
    labeler = AutoLabeler()
    print("Generating labels...")
    labels = labeler.label_documents(chunks[:10])  # 처음 10개만
    
    # 7. QA 생성
    qa_generator = QAGenerator()
    print("Generating QA pairs...")
    qa_data = qa_generator.generate_qa_batch(chunks[:5])  # 처음 5개만
    
    # 8. MongoDB 저장
    db_client = MongoDBClient(settings.mongodb_uri)
    
    for label in labels:
        db_client.insert_labels(label)
    
    for qa in qa_data:
        db_client.insert_qa_pairs(qa)
    
    db_client.close()
    
    print("Processing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents for RAG system")
    parser.add_argument("--input", default="data/raw", help="Input directory")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    
    args = parser.parse_args()
    process_documents(args.input, args.output)
