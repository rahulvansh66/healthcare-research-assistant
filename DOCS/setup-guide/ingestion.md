# Enterprise Agentic RAG (Taught in Stages)

This repository is taught **incrementally, one git branch per lesson**. Each
branch is a small, runnable step on top of the previous one.

## Lesson roadmap

| Stage | Branch | What you'll build |
|---|---|---|
| **1** | **`stage-1-ingestion`** ← you are here | Parse local documents, chunk them, embed them, and index them into a vector database |
| 2 | `stage-2-basic-rag` | A minimal FastAPI + LangGraph RAG agent — no reranking, no memory yet |
| 3 | `stage-3-rerank-memory` | Add a local semantic reranker and multi-turn conversation memory |
| 4 | `stage-4-guardrails` | Add an input safety gate that blocks off-topic and jailbreak attempts |
| 5 | `stage-5-llm-gateway` | Route all LLM calls through an LLM gateway with automatic fallback and caching |
| 6 | `stage-6-evals` | Add a RAGAS-based evaluation suite to measure the whole system |

---

## Stage 1 — Data Ingestion

Turns raw documents in `DATA/true_data/` into searchable vectors in Qdrant.
There is no API and no agent yet — just a standalone CLI pipeline.

```
DATA/true_data/*  →  loader (per file type)  →  chunker  →  processed_data/*.json
                                                          ↘  Gemini embeddings  →  Qdrant Cloud
```

### Project structure

```text
├── app/
│   ├── config.py          # Centralized environment variable management
│   ├── ingestion/
│   │   ├── loaders/       # Local parsers — PDF (pypdf/pdfplumber), HTML, TXT, DOCX/PPTX
│   │   ├── chunking/      # Paragraph-based text splitter (1500 char max)
│   │   └── processor.py   # CLI entrypoint — parse, chunk, embed, index
│   └── services/
│       └── retrieval/
│           └── embedding.py   # Gemini gemini-embedding-2-preview (3072-dim), local fallback
├── DATA/true_data/        # Sample documents (6 files)
└── processed_data/        # Auto-generated — parsed & chunked JSON output per document
```

### 1. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/) as its package manager.

```powershell
uv sync
```

`uv sync` reads `pyproject.toml` / `uv.lock`, creates a `.venv/` if one
doesn't exist, and installs the pinned dependency set. Run any command in
that environment with `uv run ...` (no manual `activate` needed) — e.g.
`uv run python -m app.ingestion.processor DATA/true_data true --wipe`.

To add a new dependency later: `uv add <package>`.

### Logfire setup

Install Python SDK and connect
Run this in your project's terminal or console to install the SDK, sign in with logfire auth (OAuth in your browser, no API key required), and connect this project.

uv add 'logfire[system-metrics]'
uv run logfire auth
uv run logfire projects use --org 'rahulvansh66' 'starter-project'
Copy


Deploying to production or running in CI? Click to reveal a write token you can use to send telemetry to Logfire.
Send telemetry directly from your code
Hello World
Pydantic AI
Anthropic
OpenAI
Save this as main.py and run it with uv run main.py.

import logfire

logfire.configure()
logfire.instrument_system_metrics()
logfire.info('Hello, {place}!', place='World')

### 2. Configure environment

fill `.env` : `QDRANT_API_KEY`,
`QDRANT_CLUSTER_ENDPOINT` (and `LOGFIRE_TOKEN` if you want tracing).

### 3. Run ingestion

```powershell
uv run python -m app.ingestion.processor DATA/true_data true --wipe
```

This parses all 6 files in `DATA/true_data/`, chunks them (1500 chars,
paragraph-aware), writes metadata to `processed_data/true/*.json`, embeds
each chunk with Gemini, and upserts the vectors into a Qdrant collection
named `enterprise_rag`. Pass `--wipe` to drop and recreate the collection;
omit it to append.

### Verify it worked

- `processed_data/true/` should contain 6 JSON files.
- The Qdrant collection's point count should equal the total chunk count
  across those 6 files (check via the Qdrant Cloud console, or
  `QdrantClient(...).count(collection_name="enterprise_rag")`).

Next: check out `stage-2-basic-rag` to turn this indexed data into an
answerable RAG agent.
