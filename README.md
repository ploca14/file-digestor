# Medical Record Extraction Service

> Built at [Hack Jak Brno 2024](https://www.hackjakbrno.cz/home-2024/), where it won the InterSystems challenge.

Turns messy medical documents (scans, PDFs, clinical notes) into structured **HL7 FHIR** data, a plain-text clinical summary, and a patient-friendly explanation — then answers questions about them with retrieval-augmented AI.

Built around **InterSystems IRIS** as the vector store, so every processed document is semantically searchable and scoped to a patient.

## What it does

- **Document processing** — parses medical documents with high-resolution layout extraction; multilingual (including Czech), preserving the original content.
- **Three outputs per document:**
  - **HL7 FHIR R4** — structured, standards-compliant clinical data
  - **Plain-text summary** — a concise clinical overview
  - **Patient-friendly explanation** — the same content in plain language
- **AI suggestions (RAG)** — context-aware answers grounded in the relevant patient documents, keeping conversation history across a doctor–patient exchange.
- **Vector storage** — documents are embedded and stored in the IRIS Vector Store, linked to patient IDs for filtered, semantic retrieval.

## How it works

```
document → Unstructured (parse) → GPT-4 (structure → FHIR + summaries)
         → embed + store in IRIS Vector Store (per patient)
         → chat endpoint: RAG over IRIS for grounded suggestions
```

- `main.py` — FastAPI app and API endpoints (`/docs` for the OpenAPI UI)
- `chains/` — the LangChain processing and RAG chains
- `models/` — Pydantic and FHIR data models
- `dependencies/` — shared clients (IRIS connection, LLM, document parsing)
- `config.py` — configuration

## Tech stack

FastAPI · OpenAI GPT-4 · LangChain · Unstructured (document parsing) · InterSystems IRIS Vector Store · Pydantic · HL7 FHIR R4 · Python 3.9+

## Running locally

**Prerequisites:** Python 3.9+, an IRIS database, OpenAI API access, Unstructured API access.

```bash
git clone git@github.com:ploca14/file-digestor.git
cd file-digestor
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

`.env`:

```env
OPENAI_API_KEY=...
UNSTRUCTURED_API_KEY=...
IRIS_VECTOR_COLLECTION_NAME=...
IRIS_CONNECTION_STRING=...
```

Run the dev server and open the interactive API docs:

```bash
fastapi dev main.py
# http://127.0.0.1:8000/docs
```
