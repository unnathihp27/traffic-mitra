# Traffic Mitra

Traffic Mitra is an educational road-safety chatbot with a responsive web interface
and a FastAPI backend. It searches local PDF and Markdown documents using Gemini
retrieval embeddings and FAISS, then sends retrieved excerpts to Google Gemini.
Answers show the excerpts and their source filenames/page numbers. When no
sufficiently relevant evidence is found, the app does not call Gemini and reports
that the local documents are insufficient.

The included source document is general educational material with an Indian
road-awareness context. It is not official legal guidance, does not claim to be
exhaustive or current, and intentionally does not state fines or specific legal
requirements.

## Requirements

- Python 3.12 for Vercel deployments; Python 3.10+ for local development.
- A Google Gemini API key from Google AI Studio to generate answers.
- Outbound internet access so the backend can call Gemini for embeddings and answers.

## Setup on Windows PowerShell

From the project directory, create a virtual environment and install development
dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
python -m pip install -r requirements-dev.txt
```

Edit `.env` and set `GOOGLE_API_KEY` to your own key. Do not commit `.env`; it is
ignored by Git. Never paste the key into source code or share it in chat/logs. The
same key is used for retrieval embeddings and answer generation.

## Build and run

Start the app locally:

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. The API creates a Gemini-embedded FAISS index on its
first question and caches it in the current process. The Vercel function stores that
index in `/tmp`; it is recreated after a cold start, so source documents remain the
durable knowledge base. The standalone `python build_knowledge_base.py` command can
still build a local index for inspection.

Optional environment settings are shown in `.env.example`. `GEMINI_MODEL` defaults to
`gemini-3.8-flash`; change it if that model is unavailable to your account, using a
model identifier currently supported for your Google Gen AI API access.

## Tests

```powershell
python -m pytest -q
```

Unit tests use deterministic local embedding fakes and do not call Gemini or spend
API credits. PDF creation/extraction tests use PyMuPDF.

## Add knowledge documents

Place `.pdf` and/or `.md` files in `data/`, then rebuild the local index with:

```powershell
python build_knowledge_base.py --force
```

PDF text is extracted page-by-page, and one-based page numbers are retained as
metadata. Markdown documents retain their filename without a page number. Scanned
image-only PDFs need OCR before they can be used; this application does not silently
claim to extract text it cannot read. The loader reports unreadable/encrypted PDFs.
Keep documents trustworthy and appropriately licensed. Retrieved document content is
treated as untrusted input by the model prompt.
New files placed in `data/` are ignored by Git by default to avoid accidentally
committing private material. The two bundled educational guide files are explicitly
allow-listed; only allow-list another source file if you have verified it is safe to
share.

## How RAG works here

1. `document_loader.py` reads and cleans PDF/Markdown text, preserving source and page.
2. `rag_pipeline.py` chunks the text with overlap and embeds it using
   `gemini-embedding-001`, with separate retrieval task types for documents and queries.
3. Normalized vectors are stored in a CPU FAISS inner-product index. A fingerprint
   detects source/model changes; metadata is stored alongside the index.
4. Each non-empty question is embedded and searched against the local index.
5. Only chunks above the configured relevance threshold are supplied to Gemini via
   `google-genai`. If no relevant chunks exist, the grounded insufficient-context
   response is returned without making a Gemini request.
6. The UI displays source filenames, page numbers when known, similarity scores, and
   expandable retrieved excerpts under **RAG Sources**.

The threshold is configurable as `MIN_RELEVANCE_SCORE`; similarity values are cosine
similarities, not probabilities. Review and tune it for a substantially different
knowledge corpus.

## Deployment

The repository includes Vercel configuration and a Python 3.12 runtime pin.

1. Import `unnathihp27/traffic-mitra` into Vercel and leave the project root at the
   repository root.
2. Add `GOOGLE_API_KEY` in **Project Settings → Environment Variables** for Production
   and Preview. Optionally add `GEMINI_MODEL`, `EMBEDDING_MODEL_NAME`, `TOP_K`, and
   `MIN_RELEVANCE_SCORE`.
3. Deploy. The static client is served from `public/`; `/api/ask` and `/api/health`
   are handled by the FastAPI function in `app.py`.
4. Open the Vercel deployment URL and check `/api/health`; it should return
   `{"status":"ok"}`. Ask a road-safety question in the chat and confirm the answer
   includes expandable source excerpts. If the API reports a missing key, add or
   correct `GOOGLE_API_KEY` in Vercel settings and redeploy.

The committed vector index uses a different embedding model and is not reused. Each
warm function instance builds a matching index in `/tmp` on its first question;
Vercel's temporary filesystem is not durable, so a cold start may repeat embedding
requests and incur Gemini usage. Larger or high-traffic deployments should move
vectors to a managed vector database. Configure request quotas or rate limits for the
public API to control abuse and API costs. Never commit `.env` or paste secrets into
source code.

## Disclaimer

Traffic Mitra is for education and general awareness only. It does not replace
official traffic regulations, emergency services, vehicle-manufacturer guidance, or
professional advice. Rules and emergency contacts vary by location and can change.
For current requirements, consult relevant official authorities; in an emergency,
contact local emergency services.
