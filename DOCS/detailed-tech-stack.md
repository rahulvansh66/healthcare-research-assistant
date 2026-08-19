
## Tech Stack

**API & Web**

- FastAPI + Uvicorn — REST API and ASGI server
- Streamlit — chat frontend and eval demo app

**Agentic Orchestration & RAG**

- LangChain + LangGraph — core LLM orchestration and cyclic multi-agent state machines
- Pydantic — data validation and settings management

**Embeddings**

- Main: `NeuML/biomedbert-small-embeddings` (22.7M params, 384-dim)
- Fallback: `NeuML/biomedbert-hash-nano-embeddings` (~1M params, 128-dim, loaded with `trust_remote_code=True`)

**Vector DB & Retrieval**

- Qdrant — vector database for semantic search
- FlashRank (TinyBERT) — fast local cross-encoder reranker (uses `sentence-transformers` internally)

**LLM Gateway**

- Portkey — unified LLM routing, fallbacks, and observability (OpenAI-compatible layer via `langchain-openai`)
- Groq — used by the guardrails classifier for low-latency inference

**Guardrails**

- NVIDIA NeMo Guardrails — input/output safety rails

**Observability & Tracing**

- LangSmith — LangChain/LangGraph tracing
- Logfire — FastAPI + requests instrumentation

**Evaluation**

- RAGAS — RAG metrics (faithfulness, relevancy, recall)
- DeepEval — pytest-compatible eval test runner
- Langfuse — production monitoring and user feedback

**Data Ingestion & Parsing**

- `unstructured`, `python-docx`, `python-pptx` — Office document parsing
- `pypdf` + `pdfplumber` (fallback for scanned/complex layouts) — PDF extraction
- `beautifulsoup4` — HTML extraction -->



