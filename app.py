"""Traffic Mitra Streamlit application."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from config import (
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    INDEX_PATH,
    METADATA_PATH,
    TOP_K,
)
from rag_pipeline import (
    GeminiConfigurationError,
    GeminiGenerationError,
    KnowledgeBaseError,
    generate_grounded_answer,
    get_embedding_model,
    knowledge_source_fingerprint,
    load_or_build_index,
    retrieve_relevant_chunks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Traffic Mitra | Road-safety assistant",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --tm-ink: #17324d;
        --tm-muted: #405569;
        --tm-border: #c7d7dc;
        --tm-teal: #087e83;
        --tm-amber: #f0ad32;
        --tm-control: #17324d;
    }
    .stApp { background: linear-gradient(180deg, #f3f8fa 0%, #ffffff 40%); }
    [data-testid="stSidebar"] {
        background: #eaf3f4;
        color: var(--tm-ink);
    }
    [data-testid="stSidebar"] :is(
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        [data-testid="stRadio"] label
    ) { color: var(--tm-ink); }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] main [data-testid="stCaptionContainer"] {
        color: var(--tm-muted);
        opacity: 1;
    }
    [data-testid="stAppViewContainer"] main :is(
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        [data-testid="stSpinner"]
    ) { color: var(--tm-ink); }
    [data-testid="stAppViewContainer"] main [data-testid="stAlert"] {
        color: var(--tm-ink);
    }
    .hero {
        padding: 1.3rem 1.5rem;
        border-radius: 18px;
        color: #ffffff;
        background: linear-gradient(115deg, #12344c, #087e83);
        margin-bottom: 1.1rem;
    }
    .hero h1 { margin: 0; color: #ffffff; font-size: 2rem; }
    .hero p { margin: .4rem 0 0; color: #e6f4f4; }
    .disclaimer { color: var(--tm-muted); font-size: .82rem; padding-top: 1rem; }
    [data-testid="stAppViewContainer"] .stButton > button {
        color: #ffffff;
        background: var(--tm-control);
        border: 1px solid var(--tm-control);
    }
    [data-testid="stAppViewContainer"] .stButton > button * {
        color: #ffffff;
    }
    [data-testid="stAppViewContainer"] .stButton > button:hover:not(:disabled) {
        color: #ffffff;
        background: var(--tm-teal);
        border-color: var(--tm-teal);
    }
    [data-testid="stAppViewContainer"] .stButton > button:focus-visible {
        outline: 3px solid var(--tm-amber);
        outline-offset: 2px;
    }
    [data-testid="stAppViewContainer"] .stButton > button:disabled {
        color: #e8eef2;
        background: #526579;
        border-color: #526579;
        opacity: 1;
    }
    [data-testid="stAppViewContainer"] main [data-testid="stChatMessage"] {
        color: var(--tm-ink);
        background: #ffffff;
        border: 1px solid var(--tm-border);
        border-radius: 12px;
    }
    [data-testid="stAppViewContainer"] main [data-testid="stExpander"] {
        color: var(--tm-ink);
        background: #ffffff;
        border: 1px solid var(--tm-border);
    }
    [data-testid="stAppViewContainer"] main [data-testid="stExpander"] summary {
        color: var(--tm-ink);
    }
    [data-testid="stChatInput"] {
        background: #1f2933;
        border: 1px solid #526579;
    }
    [data-testid="stChatInput"]:focus-within { border-color: #56c5c0; }
    [data-testid="stChatInput"] textarea {
        color: #ffffff;
        caret-color: #ffffff;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #cbd5df;
        opacity: 1;
    }
    [data-testid="stChatInput"] button { color: #ffffff; }
    [data-testid="stChatInput"] button:disabled {
        color: #f8fafc;
        background: #526579;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the local embedding model…")
def cached_embedding_model(model_name: str):
    return get_embedding_model(model_name)


@st.cache_resource(show_spinner="Preparing the local knowledge index…")
def cached_knowledge_base(
    data_dir: str,
    index_file: str,
    metadata_file: str,
    model_name: str,
    source_fingerprint: str,
):
    return load_or_build_index(
        Path(data_dir), Path(index_file), Path(metadata_file), model_name=model_name
    )


def get_api_key() -> str | None:
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GOOGLE_API_KEY")
    except (FileNotFoundError, KeyError, StreamlitSecretNotFoundError):
        return None


def submit_question(question: str) -> None:
    question = question.strip()
    if not question:
        st.warning("Please enter a question before sending.")
        return
    st.session_state.messages.append({"role": "user", "content": question})
    sources = []
    try:
        model = cached_embedding_model(EMBEDDING_MODEL_NAME)
        source_fingerprint = knowledge_source_fingerprint(
            DATA_DIR, EMBEDDING_MODEL_NAME
        )
        bundle = cached_knowledge_base(
            str(DATA_DIR),
            str(INDEX_PATH),
            str(METADATA_PATH),
            EMBEDDING_MODEL_NAME,
            source_fingerprint,
        )
        sources = retrieve_relevant_chunks(
            question, bundle, model, top_k=TOP_K
        )
        answer = generate_grounded_answer(
            question, sources, get_api_key()
        )
    except GeminiConfigurationError as exc:
        answer = str(exc)
    except GeminiGenerationError as exc:
        answer = str(exc)
    except KnowledgeBaseError as exc:
        logger.error("Knowledge base unavailable (%s)", type(exc).__name__)
        message = str(exc)
        if message.startswith("Could not read PDF '"):
            user_message = message
        elif "No readable text" in message:
            user_message = (
                "The knowledge documents contain no extractable text. Check that the "
                "PDFs are text-based (image-only scans need OCR) and Markdown files "
                "are not empty."
            )
        else:
            user_message = (
                "The local knowledge base is unavailable. Add readable PDF or Markdown "
                "files under `data/`, then rebuild the knowledge base."
            )
        answer = user_message
    except Exception as exc:
        logger.error("Question handling failed (%s)", type(exc).__name__)
        answer = "Something went wrong while processing that question. Please try again."
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
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
    )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.markdown("## 🚦 Traffic Mitra")
    st.caption("Road-safety knowledge, grounded in your local documents.")
    section = st.radio("Navigate", ["Ask Traffic Mitra", "About this assistant"])
    st.divider()
    st.markdown("### Knowledge base")
    document_count = len(
        [
            path
            for path in DATA_DIR.glob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".md"}
        ]
    )
    st.write(f"{'🟢' if document_count else '🔴'} {document_count} source document(s) found")
    if INDEX_PATH.is_file() and METADATA_PATH.is_file():
        st.caption("Saved FAISS index found (validated when used).")
    else:
        st.caption("Index will be built automatically on the first question.")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL_NAME}`")
    if st.button("Rebuild knowledge base", use_container_width=True):
        try:
            with st.spinner("Rebuilding the FAISS index…"):
                model = cached_embedding_model(EMBEDDING_MODEL_NAME)
                load_or_build_index(
                    DATA_DIR,
                    INDEX_PATH,
                    METADATA_PATH,
                    embedding_model=model,
                    model_name=EMBEDDING_MODEL_NAME,
                    force_rebuild=True,
                )
                cached_knowledge_base.clear()
            st.success("Knowledge base rebuilt.")
        except KnowledgeBaseError as exc:
            logger.error("Manual rebuild failed (%s)", type(exc).__name__)
            st.error("Could not rebuild the knowledge base. Check the source files.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

if section == "About this assistant":
    st.title("About Traffic Mitra")
    st.write(
        "Traffic Mitra searches the local traffic-safety documents, then asks Google "
        "Gemini to answer using only the retrieved evidence. Source excerpts are "
        "shown with each answer. It is an educational aid, not an official legal source."
    )
    st.info("The local FAISS index is created automatically when the app first runs.")
else:
    st.markdown(
        '<div class="hero"><h1>🚦 Traffic Mitra</h1>'
        "<p>Know the road. Make safer decisions.</p></div>",
        unsafe_allow_html=True,
    )
    st.write(
        "Ask about road signs, safe driving, or what to do in an emergency. "
        "Answers are grounded in the local knowledge documents, with sources shown below."
    )

    st.markdown("#### Try asking")
    suggestions = [
        "What does a triangular traffic sign usually indicate?",
        "What should I do if I witness a road accident?",
        "Why should I maintain a safe following distance?",
        "What should I check before a long journey?",
    ]
    cols = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        if cols[index % 2].button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources", [])
            if message["role"] == "assistant" and sources:
                st.markdown("##### RAG Sources")
                for source_index, source in enumerate(sources, start=1):
                    page = f", page {source['page']}" if source["page"] else ""
                    with st.expander(
                        f"{source_index}. {source['source']}{page} · similarity {source['score']:.2f}"
                    ):
                        st.write(source["text"])

    question_from_chat = st.chat_input("Ask a traffic-safety question…")
    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = None
    question = question_from_chat or pending
    if question:
        with st.spinner("Searching the knowledge base and preparing an answer…"):
            submit_question(question)
        st.rerun()

st.markdown(
    '<div class="disclaimer">Educational information only. This application does not '
    "replace official traffic regulations, emergency services, or professional advice. "
    "Always follow current local guidance and contact emergency services in an emergency.</div>",
    unsafe_allow_html=True,
)
