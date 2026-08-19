from typing import Literal, Optional

import logfire
from pydantic import BaseModel

from app.agents.state import AgentState
from app.gateway import get_langchain_llm

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")


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
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are Medico's Research Planner, an intelligent triage assistant for a healthcare
    research application backed by live PubMed search.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered
       using ONLY the conversation history above (e.g., "what did you just tell me"),
       set intent=CONVERSATIONAL.
    2. If it is a clinical, medical, or health-research question requiring evidence from
       trusted medical literature (e.g., drug efficacy, treatment guidelines, disease
       information), set intent=CLINICAL and:
       - search_query: a concise, PubMed-friendly rewrite of the question (plain text,
         no boolean operators — those are added downstream).
       - date_from / date_to: a publication year range if the user implies one
         (e.g. "recent studies" -> a sensible recent date_from), else leave null.
       - publication_type: a PubMed publication type filter if the user asks for a
         specific evidence quality (e.g. "randomized controlled trial", "meta-analysis"),
         else leave null.
       - fresh_search_requested: true if the latest message asks to retry, search again,
         find something better/more/different, or otherwise expresses dissatisfaction
         with the previous answer's evidence. Otherwise false.
    """

    with logfire.span("🧠 Planner Decision"):
        decision = structured_llm.invoke(prompt)
        logfire.info(f"Intent identified: {decision.intent}")

    if decision.intent == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "search_params": None,
            "fresh_search_requested": False,
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
        "status": f"Researching clinical evidence for: {decision.search_query}",
        "plan": ["Intent: Clinical Research", f"Search Term: {decision.search_query}"]
    }
