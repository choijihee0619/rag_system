import unittest
from src.data_processing.cleaner import TextCleaner
from src.data_processing.chunker import TextChunker

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.cleaner = TextCleaner()
        self.chunker = TextChunker(chunk_size=100, overlap=20)
    
    def test_text_cleaning(self):
        dirty_text = "이 기사에는 광고가 포함될 수 있습니다. <p>Hello World</p>"
        clean_text = self.cleaner.clean_text(dirty_text)
        
        self.assertNotIn("광고", clean_text)
        self.assertNotIn("<p>", clean_text)
    
    def test_text_chunking(self):
        text = "A" * 300  # 300자 텍스트
        chunks = self.chunker.split_text(text)
        
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 100)

if __name__ == "__main__":
    unittest.main()
