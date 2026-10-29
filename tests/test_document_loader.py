from pathlib import Path

import pymupdf as fitz
import pytest

from document_loader import (
    KnowledgeDocument,
    chunk_documents,
    load_documents,
)


def test_loads_markdown_and_cleans_text(tmp_path: Path):
    source = tmp_path / "guide.md"
    source.write_text("# Safety\n\n  Leave   space. \n\n\nDrive carefully.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].source == "guide.md"
    assert documents[0].page is None
    assert "Leave space." in documents[0].text
    assert "\n\n\n" not in documents[0].text


def test_loads_pdf_text_with_one_based_page_metadata(tmp_path: Path):
    pdf_path = tmp_path / "guide.pdf"
    pdf = fitz.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "First page safety text")
    second = pdf.new_page()
    second.insert_text((72, 72), "Second page helmet advice")
    pdf.save(pdf_path)
    pdf.close()

    documents = load_documents(tmp_path)

    assert [(item.source, item.page) for item in documents] == [
        ("guide.pdf", 1),
        ("guide.pdf", 2),
    ]
    assert "helmet advice" in documents[1].text


def test_chunking_overlaps_and_preserves_source_metadata():
    source = KnowledgeDocument(
        text="a" * 100 + " " + "b" * 100, source="manual.pdf", page=7
    )

    chunks = chunk_documents([source], chunk_size=80, overlap=20)

    assert len(chunks) >= 2
    assert all(chunk.source == "manual.pdf" and chunk.page == 7 for chunk in chunks)
    assert len(chunks[0].text) <= 80
    assert set(chunks[0].text[-15:]) & set(chunks[1].text[:15])


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_documents([KnowledgeDocument("text", "guide.md")], 10, 10)


def test_empty_or_missing_knowledge_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)

    (tmp_path / "blank.md").write_text(" \n ", encoding="utf-8")
    with pytest.raises(ValueError, match="No readable text"):
        load_documents(tmp_path)


def test_unreadable_pdf_reports_its_filename(tmp_path: Path):
    (tmp_path / "broken.pdf").write_bytes(b"not a PDF")

    with pytest.raises(ValueError, match="Could not read PDF 'broken.pdf'"):
        load_documents(tmp_path)
