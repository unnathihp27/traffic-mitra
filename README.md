# Traffic Mitra

Traffic Mitra is an educational road-safety chatbot built with Streamlit and a
retrieval-augmented generation (RAG) pipeline. It searches local PDF and Markdown
documents using Sentence Transformers and FAISS, then sends retrieved excerpts to
Google Gemini. Answers show the excerpts and their source filenames/page numbers.
When no sufficiently relevant evidence is found, the app does not call Gemini and
reports that the local documents are insufficient.

The included source document is general educational material with an Indian
road-awareness context. It is not official legal guidance, does not claim to be
exhaustive or current, and intentionally does not state fines or specific legal
requirements.

## Requirements

- Python 3.11 or compatible Python 3.10+.
- A Google Gemini API key from Google AI Studio to generate answers.
- Internet access on first use to download the embedding model and call Gemini.
- CPU and memory sufficient for Sentence Transformers and FAISS. The embedding model
  is downloaded once and cached by the model library.

## Setup on Windows PowerShell

From the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
$runtime = "D:\Traffic_Mitra-runtime"
New-Item -ItemType Directory -Force -Path "$runtime\pip-tmp", "$runtime\pip-cache", "$runtime\hf-cache" | Out-Null
$env:TEMP = "$runtime\pip-tmp"
$env:TMP = "$runtime\pip-tmp"
$env:PIP_CACHE_DIR = "$runtime\pip-cache"
$env:HF_HOME = "$runtime\hf-cache"
$env:SENTENCE_TRANSFORMERS_HOME = "$runtime\hf-cache\sentence-transformers"
python -m pip install -r requirements.txt
```

Edit `.env` and set `GOOGLE_API_KEY` to your own key. Do not commit `.env`; it is
ignored by Git. Never paste the key into source code or share it in chat/logs.
Alternatively, configure `GOOGLE_API_KEY` in Streamlit secrets for hosted deployments.
The PowerShell commands above place temporary installer files and downloaded model
weights outside the workspace on `D:`. Choose a drive with several gigabytes free and
update `$runtime` if the project is not on `D:`. Keep these environment variables set
in the PowerShell session when building the index or running the application so the
embedding model cache also stays on that drive.

## Build and run

Build the local index:

```powershell
python build_knowledge_base.py
```

Start the app:

```powershell
streamlit run app.py
```

The app also builds the index automatically on first use and rebuilds it when source
documents or the embedding model change. Use the sidebar's **Rebuild knowledge base**
button or run `python build_knowledge_base.py --force` to rebuild manually. Index
files and their chunk metadata are generated under `vector_store/` and are excluded
from Git.

Optional environment settings are shown in `.env.example`. `GEMINI_MODEL` defaults to
`gemini-2.5-flash`; change it if that model is unavailable to your account, using a
model identifier currently supported for your Google Gen AI API access.

## Tests

```powershell
pytest -q
```

Unit tests use deterministic local embedding fakes and do not call Gemini or spend
API credits. PDF creation/extraction tests use PyMuPDF.

## Add knowledge documents

Place `.pdf` and/or `.md` files in `data/`, then use **Rebuild knowledge base** or:

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
   `sentence-transformers/all-MiniLM-L6-v2`.
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

Use a Python-compatible host that can run Streamlit and install packages from
`requirements.txt`. Configure `GOOGLE_API_KEY` as a private environment variable or
the host's Streamlit secret; never bake it into the image or repository. If desired,
configure `GEMINI_MODEL` as an environment variable as well.

Some hosting providers use ephemeral filesystems, so do not depend on a previously
generated `vector_store/` surviving a restart or new deployment. A reliable startup
sequence installs dependencies, ensures the bundled documents are present, builds
the index, then starts Streamlit:

```powershell
python -m pip install -r requirements.txt
python build_knowledge_base.py
streamlit run app.py --server.address 0.0.0.0 --server.port $env:PORT
```

On Linux-based hosts, use the equivalent shell command:

```sh
python -m pip install -r requirements.txt
python build_knowledge_base.py
streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
```

For managed services, configure these steps as the build/start command according to
the provider's process model. Building at startup is safe: the index is reused when
valid and rebuilt when missing or stale. Confirm that the host has enough disk,
memory, outbound network access, and a writable local index directory.

## Disclaimer

Traffic Mitra is for education and general awareness only. It does not replace
official traffic regulations, emergency services, vehicle-manufacturer guidance, or
professional advice. Rules and emergency contacts vary by location and can change.
For current requirements, consult relevant official authorities; in an emergency,
contact local emergency services.
