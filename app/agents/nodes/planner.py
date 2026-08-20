import re
from typing import Literal, Optional

import logfire
from pydantic import BaseModel

from app.agents.history import format_history
from app.agents.prompts.planner import build_planner_prompt
from app.agents.state import AgentState
from app.config import settings
from app.gateway import get_langchain_llm

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")

_PUBMED_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_PMID_MENTION_RE = re.compile(r"\bPMID\s*:?\s*(\d+)\b", re.IGNORECASE)


def _detect_direct_pmid(message: str) -> Optional[str]:
    """Conservative direct-PMID detection: a PubMed URL or an explicit
    'PMID 123456' mention. Bare numeric-only messages are NOT treated as a
    PMID (too ambiguous) and fall through to the LLM planner unchanged."""
    url_match = _PUBMED_URL_RE.search(message)
    if url_match:
        return url_match.group(1)
    mention_match = _PMID_MENTION_RE.search(message)
    if mention_match:
        return mention_match.group(1)
    return None


class PlannerDecision(BaseModel):
    intent: Literal["CONVERSATIONAL", "CLINICAL"]
    search_query: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    publication_type: Optional[str] = None
    fresh_search_requested: bool = False


structured_llm = llm.with_structured_output(PlannerDecision)


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation,
    and for clinical queries extracts a PubMed-ready search query plus optional
    date-range/publication-type filters and whether the user is asking for a
    fresh search (retry/search more/not satisfied with the previous answer).
    """
    history = format_history(state["messages"][:-1], settings.MAX_HISTORY_CHARS)

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    direct_pmid = _detect_direct_pmid(user_message)
    if direct_pmid:
        logfire.info(f"Direct PMID detected: {direct_pmid}")
        return {
            "current_query": f"PMID:{direct_pmid}",
            "search_params": None,
            "fresh_search_requested": False,
            "direct_pmid": direct_pmid,
            "status": f"Fetching PMID {direct_pmid}...",
            "plan": ["Intent: Direct PMID Lookup", f"PMID: {direct_pmid}"],
        }

    prompt = build_planner_prompt(history, user_message)

    with logfire.span("🧠 Planner Decision"):
        decision = structured_llm.invoke(prompt)
        logfire.info(f"Intent identified: {decision.intent}")

    if decision.intent == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "search_params": None,
            "fresh_search_requested": False,
            "direct_pmid": None,
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision.search_query,
        "search_params": {
            "date_from": decision.date_from,
            "date_to": decision.date_to,
            "publication_type": decision.publication_type,
        },
        "fresh_search_requested": decision.fresh_search_requested,
        "direct_pmid": None,
        "status": f"Researching clinical evidence for: {decision.search_query}",
        "plan": ["Intent: Clinical Research", f"Search Term: {decision.search_query}"]
    }
