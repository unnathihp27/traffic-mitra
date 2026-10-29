from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf as fitz
import numpy as np
import pytest

from rag_pipeline import (
    INSUFFICIENT_CONTEXT,
    GeminiConfigurationError,
    GeminiEmbeddingModel,
    KnowledgeBaseError,
    answer_question,
    build_knowledge_base,
    GeminiGenerationError,
    generate_grounded_answer,
    load_index,
    load_or_build_index,
    retrieve_relevant_chunks,
)


class TinyEmbedder:
    """Deterministic tiny vectors for isolated retrieval tests."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            normalized = text.lower()
            if any(word in normalized for word in ("helmet", "head")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(word in normalized for word in ("rain", "visibility")):
                vectors.append([0.0, 1.0, 0.0])
            elif any(word in normalized for word in ("space", "travel")):
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.0, 0.0, -1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_gemini_embedder_configures_retrieval_task_and_dimension():
    client = Mock()
    client.models.embed_content.return_value.embeddings = [
        SimpleNamespace(values=[1.0, 0.0])
    ]
    embedding_model = GeminiEmbeddingModel.__new__(GeminiEmbeddingModel)
    embedding_model.model_name = "gemini-embedding-001"
    embedding_model.client = client

    vectors = embedding_model.encode_for_task(
        ["road safety"], "RETRIEVAL_DOCUMENT"
    )

    request = client.models.embed_content.call_args.kwargs
    assert request["model"] == "gemini-embedding-001"
    assert request["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert request["config"].output_dimensionality == 768
    assert vectors.shape == (1, 2)


def write_sources(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "guide.md").write_text(
        "# Road safety\n\nA fitted helmet helps protect the head.\n\n"
        "In rain, slow down and increase the following gap.",
        encoding="utf-8",
    )


def make_index(tmp_path, embedder=None):
    data_dir = tmp_path / "data"
    write_sources(data_dir)
    return build_knowledge_base(
        data_dir,
        tmp_path / "vector_store" / "test.faiss",
        tmp_path / "vector_store" / "test.json",
        embedding_model=embedder or TinyEmbedder(),
        model_name="test-model",
        chunk_size=80,
        chunk_overlap=10,
    )


def test_vector_retrieval_returns_matching_source_and_page(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "A helmet protects the head.")
    pdf.save(data_dir / "safety.pdf")
    pdf.close()
    bundle = build_knowledge_base(
        data_dir,
        tmp_path / "index.faiss",
        tmp_path / "metadata.json",
        embedding_model=TinyEmbedder(),
        model_name="test-model",
    )

    retrieved = retrieve_relevant_chunks(
        "Why use a helmet?", bundle, TinyEmbedder(), min_score=0.5
    )

    assert retrieved
    assert retrieved[0].source == "safety.pdf"
    assert retrieved[0].page == 1


def test_missing_context_does_not_call_gemini(tmp_path: Path):
    bundle = make_index(tmp_path)
    client = Mock()
    answer, sources = answer_question(
        "Tell me about space travel",
        bundle,
        TinyEmbedder(),
        api_key=None,
        min_score=0.9,
        client=client,
    )

    assert answer == INSUFFICIENT_CONTEXT
    assert sources == []
    client.models.generate_content.assert_not_called()


def test_gemini_generation_uses_mocked_client_and_source_citation(tmp_path: Path):
    bundle = make_index(tmp_path)
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text="A helmet helps protect the head [Source 1: guide.md]."
    )

    answer, sources = answer_question(
        "Why wear a helmet?",
        bundle,
        TinyEmbedder(),
        api_key="unit-test-key",
        min_score=0.1,
        client=client,
    )

    assert "Source 1" in answer
    assert sources
    client.models.generate_content.assert_called_once()


def test_gemini_rate_limit_is_explained_without_leaking_details():
    client = Mock()
    client.models.generate_content.side_effect = RuntimeError("private diagnostic detail")
    client.models.generate_content.side_effect.code = 429
    from rag_pipeline import RetrievedChunk

    with pytest.raises(GeminiGenerationError, match="rate limit") as exc_info:
        generate_grounded_answer(
            "Question?",
            [RetrievedChunk("Evidence", "guide.md", None, 0.9)],
            api_key="unit-test-key",
            client=client,
        )

    assert "private diagnostic detail" not in str(exc_info.value)


def test_missing_api_key_is_clear_when_context_exists(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    bundle = make_index(tmp_path)
    with pytest.raises(GeminiConfigurationError):
        answer_question(
            "Why wear a helmet?",
            bundle,
            TinyEmbedder(),
            api_key=None,
            min_score=0.1,
        )


def test_index_reuse_and_source_change_rebuild(tmp_path: Path):
    data_dir = tmp_path / "data"
    write_sources(data_dir)
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.json"
    first = load_or_build_index(
        data_dir, index_path, metadata_path, TinyEmbedder(), model_name="test-model"
    )
    reused = load_or_build_index(
        data_dir, index_path, metadata_path, TinyEmbedder(), model_name="test-model"
    )
    assert reused.fingerprint == first.fingerprint
    assert reused.index.ntotal == first.index.ntotal

    (data_dir / "guide.md").write_text("Helmet head protection and safe riding.", encoding="utf-8")
    rebuilt = load_or_build_index(
        data_dir, index_path, metadata_path, TinyEmbedder(), model_name="test-model"
    )
    assert rebuilt.fingerprint != first.fingerprint


def test_corrupt_index_rebuilds_and_invalid_load_fails(tmp_path: Path):
    bundle = make_index(tmp_path)
    data_dir = tmp_path / "data"
    index_path = tmp_path / "vector_store" / "test.faiss"
    metadata_path = tmp_path / "vector_store" / "test.json"
    index_path.write_bytes(b"not a FAISS index")

    rebuilt = load_or_build_index(
        data_dir,
        index_path,
        metadata_path,
        TinyEmbedder(),
        model_name="test-model",
    )

    assert rebuilt.index.ntotal == len(rebuilt.chunks)
    assert load_index(index_path, metadata_path, model_name="test-model").chunks

    metadata_path.write_text("invalid json", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError):
        load_index(index_path, metadata_path, model_name="test-model")
