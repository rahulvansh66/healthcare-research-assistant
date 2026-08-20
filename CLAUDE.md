# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@.claude/git-conventions.md

## Commands

```bash
# Install deps (preferred)
uv sync

# Configure
cp .env.example .env   # fill in credentials — see comments in .env.example
# NCBI_CONTACT_EMAIL is required (NCBI usage policy); NCBI_API_KEY is optional
# (raises the E-utilities rate limit from 3 req/s to 10 req/s)

# Run the backend — no ingestion step needed, retrieval is live against PubMed
uv run uvicorn app.main:app --reload

# Run the chat UI (separate terminal)
uv run streamlit run streamlit_app.py

# Run the eval suite UI (backend must be running on :8000)
uv run streamlit run evals/app.py
```

There is no automated test suite in this repo — correctness is checked via the eval suite (`evals/`), which drives the live `/query` endpoint against `evals/golden_dataset.json` and scores with RAGAS/DeepEval.

## Architecture

Request flow: `Streamlit UI` → `FastAPI /query` → `NeMo Guardrails gate` → `LangGraph agent` → response.

- **Guardrails gate first** (`app/guardrails/rails.py`, rules in `colang_rules.py`): every query hits this before the agent. If a rail fires, `main.py` returns the guardrail's response immediately and the LangGraph agent never runs. Uses a small/fast Groq model (`allam-2-7b`) for intent classification, distinct from the main RAG model. Layers: a regex-based PII check (`app/guardrails/pii.py` — email/phone/SSN/credit-card patterns, checked in plain Python before the NeMo call; deliberately not an NLP/NER model like Presidio+spaCy, which needs a large model resident in memory per request), few-shot dialog rails (off-topic/jailbreak/greeting/farewell/capabilities), and a `self check input` LLM policy check (catches jailbreak paraphrases the few-shot examples miss). This is an **input-only** gate — the LangGraph-generated answer is never routed back through NeMo, so there's no equivalent output rail.
- **Input validation & rate limiting** (`app/main.py`): `QueryRequest.q`/`thread_id` are length-bounded via Pydantic `Field` constraints (`settings.MAX_QUERY_CHARS`/`MAX_THREAD_ID_CHARS`), rejecting empty/whitespace-only or oversized input with a 422 before the guardrails gate even runs. `POST /query` is rate-limited per client IP via `slowapi` (`settings.RATE_LIMIT_QUERY`). Conversation history inlined into planner/responder prompts is capped to `settings.MAX_HISTORY_CHARS` via the shared `app/agents/history.format_history()` helper (drops oldest turns first) so prompt size stays bounded regardless of conversation length.
- **LangGraph agent** (`app/agents/graph.py`): a 4-node graph — `planner` → (conditionally) `retriever` → `evidence_agent` → `responder` → END, compiled with `MemorySaver` checkpointed by `thread_id` for per-conversation memory. State shape is `AgentState` (`app/agents/state.py`).
  - `planner` (`nodes/planner.py`) reads the full conversation history and, via structured output (`.with_structured_output()`), classifies the latest message as `CONVERSATIONAL` (routes straight to responder, uses memory only) or `CLINICAL` — extracting a rewritten PubMed search query, optional date-range/publication-type filters, and whether the user is asking for a fresh search (`fresh_search_requested`, e.g. "search more"/"try again").
  - `retriever` (`nodes/retriever.py`) decides between a live PubMed search and this thread's cached results: a fresh live search runs on the first message of a conversation, when the UI's "PubMed Search" toggle is on, or when the planner detected `fresh_search_requested`; otherwise it queries the thread-scoped session cache first and only falls through to a live search if that cache is empty or below a relevance threshold. Live search calls NCBI E-utilities (`services/retrieval/pubmed_service.py`: ESearch → batched EFetch), reranks locally with FlashRank (`services/retrieval/ranking_service.py`), keeps the top `PUBMED_STORE_TOP_N` for the session cache and the top `EVIDENCE_TOP_N` for this turn.
  - `evidence_agent` (`nodes/evidence.py`) turns retrieved PubMed abstracts into structured, PMID-grounded `Evidence` records via structured output — any record citing a PMID outside the retrieved set is dropped as a hallucination guard. Sets `evidence: []` when nothing retrieved sufficiently answers the question.
  - `responder` (`nodes/responder.py`) generates the grounded answer — either from conversation memory, an "insufficient evidence" message, or evidence-grounded prose citing PMIDs — and derives the `citations` list deterministically from `evidence` + `documents` rather than trusting the LLM to reproduce it.
  - After the graph returns, `main.py` schedules a `BackgroundTasks` job to embed and upsert this turn's live PubMed results into the session cache (Qdrant, filtered by `thread_id`) — the response is sent before that storage completes.
- **LLM gateway** (`app/gateway/client.py`): all agent nodes get their LLM via `get_langchain_llm()`, which returns a `ChatOpenAI` pointed at Portkey's OpenAI-compatible gateway (`PORTKEY_GATEWAY_URL`) rather than `ChatGroq` directly — this is what gives fallback (primary → fallback virtual key), caching, and retry across the whole agent. Do not swap this for a direct `ChatGroq` call in agent nodes; that would bypass the gateway's fallback/cache/retry behavior. Portkey is configured as a saved server-side Config (`settings.PORTKEY_CONFIG`) because this org has `block_inline_config` enabled — inline strategy/cache/retry/target dicts passed from code are rejected, so gateway behavior changes must be made in the Portkey dashboard, not in `client.py`.
- **PubMed retrieval** (`app/services/retrieval/pubmed_service.py`): live NCBI E-utilities client (ESearch + batched EFetch, `xml.etree.ElementTree` parsing, throttled to NCBI's rate limits). There is no static ingestion pipeline or pre-populated corpus — every clinical answer is grounded in literature fetched at query time or reused from this thread's session cache.
- **Session cache** (`app/services/retrieval/qdrant_service.py`): Qdrant repurposed as a per-thread cache of live PubMed results (not a static admin-curated corpus) — one shared collection (`pubmed_session_cache`) filtered by a `thread_id` payload field. `app/services/retrieval/embedding.py` (local `NeuML/biomedbert-small-embeddings`) embeds abstracts for storage and query.
- **Config** (`app/config.py`): a single `Settings` class reads all env vars (Qdrant, NCBI/PubMed, Groq, Portkey, LangSmith) via `dotenv`; import `settings` from here rather than reading `os.getenv` directly elsewhere. Secrets (`NCBI_API_KEY`, `NCBI_CONTACT_EMAIL`, etc.) are read from `.env` only — never hardcode a default for anything secret.
- **Observability**: `app/main.py` configures `logfire` before any other import (module-level ordering matters — it must capture spans from everything imported after it) so its `logfire.configure()` call must stay the first statement in any new entrypoint. LangSmith tracing is enabled globally via env vars set at the bottom of `app/config.py`.
- **Evals** (`evals/`): `pipeline.py` replays `golden_dataset.json` against the live `/query` endpoint (`app.py` is the Streamlit driver), `guardrails_eval.py` scores the guardrails gate separately, and `metrics.py` runs RAGAS/DeepEval scoring on the captured results. Since retrieval is now live, eval runs make real NCBI calls per clinical sample in addition to Groq calls — set `NCBI_API_KEY` in the eval environment to stay comfortably within NCBI's rate limit.
