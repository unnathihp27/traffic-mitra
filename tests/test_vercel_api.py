from fastapi.testclient import TestClient

import app as api
from rag_pipeline import RetrievedChunk

client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_question_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    response = client.post("/api/ask", json={"question": "How do I stay safe?"})

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_question_returns_grounded_answer_and_sources(monkeypatch):
    source = RetrievedChunk(
        text="Wear a properly fitted helmet.",
        source="guide.md",
        page=None,
        score=0.91,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(api, "get_knowledge_resources", lambda: (object(), object()))
    monkeypatch.setattr(api, "retrieve_relevant_chunks", lambda *args, **kwargs: [source])
    monkeypatch.setattr(
        api,
        "generate_grounded_answer",
        lambda *args, **kwargs: "A fitted helmet helps protect you [Source 1: guide.md].",
    )

    response = client.post("/api/ask", json={"question": "Why wear a helmet?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "A fitted helmet helps protect you [Source 1: guide.md].",
        "sources": [
            {
                "text": "Wear a properly fitted helmet.",
                "source": "guide.md",
                "page": None,
                "score": 0.91,
            }
        ],
    }


def test_question_rejects_blank_input():
    response = client.post("/api/ask", json={"question": "   "})

    assert response.status_code == 422
