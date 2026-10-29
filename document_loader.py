"""Load and chunk local PDF and Markdown knowledge documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz


@dataclass(frozen=True)
class KnowledgeDocument:
    text: str
    source: str
    page: int | None = None


@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str
    page: int | None = None


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving useful paragraph boundaries."""
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: str | Path) -> list[KnowledgeDocument]:
    """Read Markdown and PDF files in a directory, retaining PDF page numbers."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge directory does not exist: {directory}")

    documents: list[KnowledgeDocument] = []
    supported_files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".md"}
    )
    if not supported_files:
        raise FileNotFoundError(f"No PDF or Markdown knowledge documents in {directory}")

    for path in supported_files:
        if path.suffix.lower() == ".md":
            text = clean_text(path.read_text(encoding="utf-8"))
            if text:
                documents.append(KnowledgeDocument(text=text, source=path.name))
            continue

        try:
            with fitz.open(path) as pdf:
                if pdf.is_encrypted:
                    raise ValueError("PDF is encrypted")
                for page_number, page in enumerate(pdf, start=1):
                    text = clean_text(page.get_text("text"))
                    if text:
                        documents.append(
                            KnowledgeDocument(
                                text=text, source=path.name, page=page_number
                            )
                        )
        except Exception as exc:
            raise ValueError(f"Could not read PDF '{path.name}': {exc}") from exc

    if not documents:
        raise ValueError(f"No readable text found in knowledge documents in {directory}")
    return documents


def chunk_documents(
    documents: list[KnowledgeDocument],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[TextChunk]:
    """Split source documents into bounded, overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    step = chunk_size - overlap
    for document in documents:
        text = clean_text(document.text)
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind(" ", start, end),
                )
                if boundary > start + chunk_size // 2:
                    end = boundary + (1 if text[boundary] == "\n" else 0)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text, source=document.source, page=document.page
                    )
                )
            if end >= len(text):
                break
            start = max(start + 1, end - overlap if end > start + overlap else start + step)
    return chunks
