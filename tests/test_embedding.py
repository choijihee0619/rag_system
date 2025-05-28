import unittest
from unittest.mock import Mock, patch
from src.embedding.embedder import Embedder

class TestEmbedding(unittest.TestCase):
    @patch('src.embedding.embedder.OpenAIEmbeddings')
    def test_embed_query(self, mock_embeddings):
        # Mock 설정
        mock_instance = Mock()
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_instance
        
        # 테스트
        embedder = Embedder()
        result = embedder.embed_query("test query")
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result, [0.1, 0.2, 0.3])

if __name__ == "__main__":
    unittest.main()
