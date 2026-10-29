"""Traffic Mitra API for Vercel's Python runtime."""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import DATA_DIR, EMBEDDING_MODEL_NAME, GEMINI_MODEL, MIN_RELEVANCE_SCORE, TOP_K
from rag_pipeline import (
    GeminiConfigurationError,
    GeminiGenerationError,
    KnowledgeBaseError,
    generate_grounded_answer,
    get_embedding_model,
    load_or_build_index,
    retrieve_relevant_chunks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
INDEX_DIR = Path(tempfile.gettempdir()) / "traffic-mitra"
app = FastAPI(title="Traffic Mitra", docs_url=None, redoc_url=None)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@lru_cache(maxsize=1)
def get_knowledge_resources() -> tuple[Any, Any]:
    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
    bundle = load_or_build_index(
        DATA_DIR,
        INDEX_DIR / "traffic_safety.faiss",
        INDEX_DIR / "traffic_safety.metadata.json",
        embedding_model=embedding_model,
        model_name=EMBEDDING_MODEL_NAME,
    )
    return bundle, embedding_model


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask")
def ask_question(request: QuestionRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Enter a question first.")
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="The assistant is not configured yet. Add GOOGLE_API_KEY in Vercel project settings.",
        )

    try:
        bundle, embedding_model = get_knowledge_resources()
        sources = retrieve_relevant_chunks(
            question,
            bundle,
            embedding_model,
            top_k=TOP_K,
            min_score=MIN_RELEVANCE_SCORE,
        )
        answer = generate_grounded_answer(
            question,
            sources,
            os.getenv("GOOGLE_API_KEY"),
            model_name=GEMINI_MODEL,
        )
    except GeminiConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeminiGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except KnowledgeBaseError as exc:
        logger.error("Knowledge base unavailable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="The road-safety knowledge base is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.error("Question handling failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="Something went wrong while preparing an answer. Please try again.",
        ) from exc

    return {
        "answer": answer,
        "sources": [
            {
                "text": source.text,
                "source": source.source,
                "page": source.page,
                "score": source.score,
            }
            for source in sources
        ],
    }


@app.get("/{asset_path:path}", include_in_schema=False)
def serve_frontend(asset_path: str = "") -> FileResponse:
    if asset_path not in {"", "index.html"}:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(ROOT_DIR / "public" / "index.html")
