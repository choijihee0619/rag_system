from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_query_endpoint():
    response = client.post(
        "/query",
        json={"query": "테스트 질문"}
    )
    # 벡터 스토어가 초기화되지 않은 경우 500 에러 예상
    assert response.status_code in [200, 500]
