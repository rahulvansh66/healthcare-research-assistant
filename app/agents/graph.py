from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from app.agents.state import AgentState
from app.config import settings
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.evidence_extractor import evidence_extractor_node
from app.agents.nodes.evidence_validator import evidence_validator_node
from app.agents.nodes.query_rewriter import query_rewriter_node
from app.agents.nodes.web_search import web_search_node
from app.agents.nodes.responder import generate_node, is_grounded_answer
from app.agents.nodes.claim_verifier import claim_verifier_node


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("evidence_extractor", evidence_extractor_node)
workflow.add_node("evidence_validator", evidence_validator_node)
workflow.add_node("query_rewriter", query_rewriter_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("responder", generate_node)
workflow.add_node("claim_verifier", claim_verifier_node)


# 3. Routing logic — every branch is an explicit conditional edge keyed off state.

def route_planner(state: AgentState) -> str:
    """CONVERSATIONAL queries answer from memory; everything else retrieves."""
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"


def route_validator(state: AgentState) -> str:
    """
    Turns the evidence_validator's typed verdict into a routing decision:
      - direct-PMID lookups never go to web search (answer / insufficient only)
      - sufficient                                   -> responder
      - query_failure / population_mismatch + budget -> query_rewriter (retry cycle)
      - corpus_failure, or retry budget spent        -> web_search
      - anything else                                -> responder (insufficient path)
    """
    if state.get("direct_pmid"):
        return "responder"

    verdict = state.get("evidence_verdict") or {}
    if verdict.get("sufficient"):
        return "responder"

    reason = verdict.get("reason")
    retriable = reason in ("query_failure", "population_mismatch")

    if retriable and state.get("retrieval_attempts", 0) < settings.CRAG_MAX_RETRIES:
        return "query_rewriter"
    if reason == "corpus_failure" or retriable:
        return "web_search"
    return "responder"


def route_responder(state: AgentState) -> str:
    """Grounded answers get fact-checked; conversational / web-fallback /
    insufficient answers are terminal."""
    return "claim_verifier" if is_grounded_answer(state) else END


def route_claim_verifier(state: AgentState) -> str:
    """Loop back to the responder for a bounded regeneration, or finish."""
    return "responder" if state.get("needs_regeneration") else END


workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner", route_planner, {"retriever": "retriever", "responder": "responder"}
)
workflow.add_edge("retriever", "evidence_extractor")
workflow.add_edge("evidence_extractor", "evidence_validator")
workflow.add_conditional_edges(
    "evidence_validator",
    route_validator,
    {
        "responder": "responder",
        "query_rewriter": "query_rewriter",
        "web_search": "web_search",
    },
)
workflow.add_edge("query_rewriter", "retriever")
workflow.add_edge("web_search", "responder")
workflow.add_conditional_edges(
    "responder", route_responder, {"claim_verifier": "claim_verifier", END: END}
)
workflow.add_conditional_edges(
    "claim_verifier", route_claim_verifier, {"responder": "responder", END: END}
)


# --- MEMORY UPGRADE ---
# PostgresSaver persists conversations by 'thread_id' so history (and the
# sessions list) survives backend restarts. from_conn_string() is a
# contextmanager; entered manually (never exited) so the pooled connection
# lives for the process's lifetime, matching this module-level singleton.
_checkpointer_cm = PostgresSaver.from_conn_string(settings.DATABASE_URL)
checkpointer = _checkpointer_cm.__enter__()
checkpointer.setup()


# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)
