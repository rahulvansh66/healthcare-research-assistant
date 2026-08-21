# Clinical Research Assistant

An AI-powered research assistant that helps users explore **clinical research literature** by retrieving relevant articles from **PubMed** and generating evidence-grounded answers to research questions.

![Clinical Research Assistant Demo](DOCS/demo/medico.png)

## What It Can Do

- Retrieve relevant research articles from PubMed based on the user's query.
- Answer clinical research questions using information from the retrieved literature.
- Generate concise summaries of specific research papers.
- Provide responses grounded in the retrieved scientific literature.
- Keep multiple conversations going and switch between them from the sidebar, with history persisted in Postgres across backend restarts.



## Example Queries

**Summarize a specific paper**

> Generate a summary for this paper:
> [https://pubmed.ncbi.nlm.nih.gov/39796530/](https://pubmed.ncbi.nlm.nih.gov/39796530/)

**Ask a research question**

> How does the Tang cell count correlate with COVID-19 disease severity?



## Tech Stack


| Layer                          | Technology                                        |
| ------------------------------ | ------------------------------------------------- |
| API & UI                       | FastAPI, Streamlit                                |
| Agent Orchestration            | LangChain, LangGraph                              |
| Literature Retrieval           | Live PubMed (NCBI E-utilities — ESearch + EFetch) |
| Embeddings                     | Jina Embeddings API                               |
| Session Cache / Vector Search  | Qdrant (per-thread PubMed result cache)           |
| Reranker                       | Jina Reranker API                                 |
| LLM Gateway                    | Portkey + Groq                                    |
| Query Cache                    | Portkey                                           |
| Conversation Memory / Sessions | Postgres (LangGraph checkpoints + sessions list)  |
| Safety                         | NVIDIA NeMo Guardrails, regex-based PII filter    |
| Observability                  | LangSmith, Logfire                                |
| Evaluation                     | RAGAS, DeepEval                                   |


---



## Agentic AI workflow

```mermaid
graph TD
    User((User)) --> UI[Streamlit UI]
    UI --> API[FastAPI /query]
    API --> PII{PII Check}
    PII -->|Blocked| Response[Response to User]
    PII -->|Pass| Guard{NeMo Guardrails}
    Guard -->|Blocked| Response
    Guard -->|Pass| Planner{Planner Node}
    Planner -->|Conversational| Responder[Responder Node]
    Planner -->|Clinical| Retriever[Retriever Node]
    Retriever -->|Fresh search| PubMed[(Live PubMed\nESearch + EFetch)]
    Retriever -->|Cached| SessionCache[(Qdrant Session Cache)]
    PubMed --> Reranker[Jina Reranker API]
    Reranker -->|Relevant| SessionCache
    Reranker -->|Below threshold, retries left| Rewrite[CRAG Query Rewrite]
    Rewrite --> PubMed
    Reranker --> Evidence[Evidence Agent]
    SessionCache --> Evidence
    Evidence --> Responder
    Responder --> Critique{Self-Critique}
    Critique -->|Unsupported claims| Caveat[Append Self-Check Note]
    Critique -->|Supported| Response
    Caveat --> Response
    Responder -.-> Memory[(Postgres\nLangGraph Checkpoints)]
```



Each query first passes a regex-based PII check and the NeMo Guardrails gate (off-topic/jailbreak/self-check) before reaching the LangGraph agent.

 The `planner` classifies the message as conversational or clinical; for clinical queries the `retriever` runs a live PubMed search (or reuses this thread's cached results) and reranks with the Jina Reranker API, if the top reranked result is below a relevance threshold, it's treated as a **Corrective-RAG (CRAG)** miss: the query is rewritten and the search retried, up to a bounded number of attempts, before falling through to whatever is best-available. 

The `evidence_agent` then extracts PMID-grounded evidence, and the `responder` generates a citation-backed answer and **self-critiques** it against that evidence, if it finds claims the evidence doesn't support, it deterministically appends a visible "Self-Check Note" caveat rather than silently rewriting the answer. 

```
...normal answer text about the drug's efficacy...

⚠️ Self-Check Note
The following statement(s) may not be fully supported by the retrieved evidence:
- Metformin reduces cardiovascular mortality by 30% in non-diabetic patients
```

This matters in a clinical-research context: a silently corrected or dropped claim would hide the exact statement a user shouldn't act on, whereas a visible caveat lets them judge the flagged claim themselves.  

## Setup



### Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose (for Postgres)



### Configuration

```bash
cp .env.example .env
```

Fill in `.env` with your own credentials — see the comments in `.env.example` for what each variable is for. `NCBI_CONTACT_EMAIL` is required; Qdrant, Groq, Portkey, and Jina keys come from those services' own dashboards (Qdrant can be a free Cloud cluster — there's no local/self-hosted option in this setup).

### Install

```bash
# Preferred: uv resolves and installs from pyproject.toml / uv.lock
uv sync

# Fallback: pip with the exported lockfile (requirements.txt is auto-generated
# via `uv export --format requirements.txt --no-hashes -o requirements.txt`)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the App

```bash
# 1. Start Postgres
docker compose up -d postgres

# 2. Start the FastAPI backend
uv run uvicorn app.main:app --reload

# 3. In a separate terminal, start the Streamlit chat UI
uv run streamlit run streamlit_app.py
```

**Backend** — `http://127.0.0.1:8000`


| Method | Path     | Description                                              |
| ------ | -------- | -------------------------------------------------------- |
| `GET`  | `/`      | Health check                                             |
| `GET`  | `/graph` | LangGraph agent diagram                                  |
| `POST` | `/query` | Submit a query — body `{"q": "...", "thread_id": "..."}` |


**Frontend** — `http://127.0.0.1:8501`

Chat UI that calls the backend at `BACKEND_URL`, with per-conversation memory via `thread_id` and expandable sources/thought-process panels per answer.

## Running Tests

Unit and integration tests live under `tests/` and run via `pytest`.

```bash
uv run pytest
```



## Running Evals

Correctness of retrieval and generation is checked via the eval suite in `evals/`, which drives the live `/query` endpoint against `evals/golden_dataset.json` and scores with RAGAS/DeepEval. The backend must be running (`uv run uvicorn app.main:app --reload`) before starting the eval UI.

```bash
uv run streamlit run evals/app.py
```

Guardrails behavior (the input gate, independent of retrieval/generation quality) is scored separately via `evals/guardrails_eval.py`.