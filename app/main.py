# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import sys

# Windows consoles default to a cp1252 codepage, which can't encode the emoji
# used in log messages throughout this codebase and crashes logfire's console
# exporter on every span. Force UTF-8 before logfire ever prints anything.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
import json

from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agents.graph import rag_agent
from app.config import settings
from app.guardrails import initialize_rails, guard
from app.services.retrieval.qdrant_service import store_session_results
from app.services import sessions_service

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")
logfire.instrument_fastapi(app)
logfire.instrument_requests()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
def startup_event():
    initialize_rails()
    sessions_service.init_sessions_table()

class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_CHARS)
    thread_id: Optional[str] = Field(default="default_user", max_length=settings.MAX_THREAD_ID_CHARS, pattern=r"^[A-Za-z0-9_-]+$")
    pubmed_search_requested: bool = False

    @field_validator("q")
    @classmethod
    def q_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("q must not be blank")
        return v


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.get("/sessions")
def list_sessions():
    """Sessions list for the sidebar, most recently active first."""
    return sessions_service.list_sessions()


@app.get("/sessions/{thread_id}/history")
def get_session_history(thread_id: str):
    """Replays a thread's messages from the LangGraph checkpoint so the UI
    can rehydrate a past conversation when the user switches to it. Only
    role/content are available per historical turn — citations and
    thought-process are only ever computed for the turn just answered, not
    persisted per message, so they're empty for replayed history."""
    config = {"configurable": {"thread_id": thread_id}}
    state = rag_agent.get_state(config)
    messages = (state.values or {}).get("messages", []) if state else []
    return {"thread_id": thread_id, "messages": messages}


def _build_initial_state(q: str, body: QueryRequest) -> dict:
    return {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "search_params": None,
        "pubmed_search_requested": body.pubmed_search_requested,
        "fresh_search_requested": False,
        "direct_pmid": None,
        "retrieval_source": "none",
        "retrieval_attempts": 0,
        "rerank_top_score": 0.0,
        "documents": [],
        "evidence": [],
        "evidence_verdict": None,
        "citations": [],
        "web_results": None,
        "used_web_fallback": False,
        "verify_attempts": 0,
        "unsupported_claims": [],
        "needs_regeneration": False,
        "_background_store_payload": None,
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }


@app.post("/query")
@limiter.limit(settings.RATE_LIMIT_QUERY)
def query(request: Request, body: QueryRequest, background_tasks: BackgroundTasks):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = body.q
    thread_id = body.thread_id

    initial_state = _build_initial_state(q, body)

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    sessions_service.record_turn(thread_id, q)

    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "citations": [],
                "web_sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)

        if final_output.get("retrieval_source") == "live_pubmed":
            background_tasks.add_task(
                store_session_results,
                thread_id=thread_id,
                query=q,
                documents=final_output.get("_background_store_payload") or [],
            )

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
            "citations": final_output.get("citations", []),
            "web_sources": final_output.get("web_results") or []
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
            "citations": []
        }


@app.post("/query/stream")
@limiter.limit(settings.RATE_LIMIT_QUERY)
def query_stream(request: Request, body: QueryRequest):
    """
    Same pipeline as /query, but streams the final answer token-by-token as
    newline-delimited JSON: {"type": "token", ...} lines while the answer is
    generated, followed by one {"type": "done", ...} line with status/sources/
    citations, or {"type": "error", ...} on failure.
    """
    q = body.q
    thread_id = body.thread_id

    initial_state = _build_initial_state(q, body)
    config = {"configurable": {"thread_id": thread_id}}
    background_tasks = BackgroundTasks()
    sessions_service.record_turn(thread_id, q)

    def event_stream():
        try:
            # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
            rail_fired, rail_response = guard(q)
            if rail_fired:
                logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
                yield json.dumps({"type": "token", "content": rail_response}) + "\n"
                yield json.dumps({
                    "type": "done",
                    "status": "Blocked by guardrails.",
                    "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                    "sources": [],
                    "citations": [],
                    "web_sources": []
                }) + "\n"
                return

            # Gate 2: LangGraph RAG pipeline, streamed
            final_output = None
            for stream_mode, chunk in rag_agent.stream(
                initial_state, config=config, stream_mode=["custom", "values"]
            ):
                if stream_mode == "custom":
                    yield json.dumps({"type": "token", "content": chunk}) + "\n"
                else:
                    final_output = chunk

            final_output = final_output or {}

            if final_output.get("retrieval_source") == "live_pubmed":
                background_tasks.add_task(
                    store_session_results,
                    thread_id=thread_id,
                    query=q,
                    documents=final_output.get("_background_store_payload") or [],
                )

            yield json.dumps({
                "type": "done",
                "status": final_output.get("status"),
                "thought_process": final_output.get("plan"),
                "sources": final_output.get("documents", []),
                "citations": final_output.get("citations", []),
                "web_sources": final_output.get("web_results") or []
            }) + "\n"
        except Exception as e:
            logfire.error(f"❌ Backend Execution Failed: {e}")
            yield json.dumps({
                "type": "error",
                "message": "I apologize, but I encountered an internal error while processing your request. Please try again later."
            }) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        background=background_tasks,
    )
