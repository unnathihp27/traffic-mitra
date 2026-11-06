from fastapi.testclient import TestClient
import pytest

import app as api
from rag_pipeline import RetrievedChunk

client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "gemini"}


def test_question_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    response = client.post(
        "/api/ask", json={"question": "How do I stay safe while driving?"}
    )

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


def test_temporary_local_mode_answers_without_generation(monkeypatch):
    source = RetrievedChunk(
        text="Wear a properly fitted helmet.",
        source="guide.md",
        page=2,
        score=0.91,
    )
    monkeypatch.setattr(api, "TEMPORARY_LOCAL_MODE", True)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(api, "get_knowledge_resources", lambda: (object(), object()))
    monkeypatch.setattr(api, "retrieve_relevant_chunks", lambda *args, **kwargs: [source])
    monkeypatch.setattr(
        api,
        "generate_grounded_answer",
        lambda *args, **kwargs: pytest.fail("Generation must not run in local mode"),
    )

    response = client.post("/api/ask", json={"question": "Why should I wear a helmet?"})

    assert response.status_code == 200
    assert "Wear a properly fitted helmet." in response.json()["answer"]
    assert "[Source: guide.md, page 2]" in response.json()["answer"]
    assert response.json()["sources"][0]["source"] == "guide.md"


def test_gemini_generation_failure_returns_local_fallback(monkeypatch):
    source = RetrievedChunk("Use a seat belt.", "guide.md", None, 0.9)
    monkeypatch.setattr(api, "TEMPORARY_LOCAL_MODE", False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(api, "get_knowledge_resources", lambda: (object(), object()))
    monkeypatch.setattr(api, "retrieve_relevant_chunks", lambda *args, **kwargs: [source])
    monkeypatch.setattr(
        api,
        "generate_grounded_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            api.GeminiGenerationError("Gemini unavailable")
        ),
    )

    response = client.post("/api/ask", json={"question": "Should I wear a seat belt?"})

    assert response.status_code == 200
    assert "Use a seat belt." in response.json()["answer"]


def test_temporary_local_mode_health_status(monkeypatch):
    monkeypatch.setattr(api, "TEMPORARY_LOCAL_MODE", True)

    assert client.get("/api/health").json() == {
        "status": "ok",
        "mode": "temporary_local",
    }


def test_unrelated_question_is_rejected(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        api,
        "get_knowledge_resources",
        lambda: pytest.fail("Unrelated questions must not trigger retrieval"),
    )

    response = client.post("/api/ask", json={"question": "How do I bake a cake?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": api.OUT_OF_SCOPE_ANSWER,
        "sources": [],
    }


def test_no_relevant_passages_reports_guide_limitation(monkeypatch):
    monkeypatch.setattr(api, "TEMPORARY_LOCAL_MODE", True)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(api, "get_knowledge_resources", lambda: (object(), object()))
    monkeypatch.setattr(api, "retrieve_relevant_chunks", lambda *args, **kwargs: [])

    response = client.post("/api/ask", json={"question": "How do I use a helmet?"})

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "That information is not available in the current road-safety guide."
    )
    assert response.json()["sources"] == []
