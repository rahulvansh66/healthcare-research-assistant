import logfire

from app.agents.state import AgentState
from app.services.websearch import search as tavily_search


def web_search_node(state: AgentState) -> dict:
    """
    Corpus-failure fallback. Reached from route_validator when the evidence
    validator decides the question can't be answered from PubMed. Runs a general
    web search (Tavily) and hands the hits to the responder, which is required to
    answer with an explicit "outside the curated PubMed evidence base" disclaimer
    plus source links.

    Always sets ``used_web_fallback=True`` (even on an empty result set or a
    missing API key) so the responder takes the fallback branch rather than
    trying to ground an answer in absent evidence.
    """
    query = state["current_query"]

    with logfire.span("🌐 Web Search Fallback"):
        results = tavily_search(query)

    logfire.info(f"Web fallback returned {len(results)} result(s).")

    return {
        "web_results": results,
        "used_web_fallback": True,
        "status": (
            f"PubMed had no sufficient evidence — pulled {len(results)} web result(s)."
            if results
            else "PubMed had no sufficient evidence and the web fallback found nothing."
        ),
        "plan": state["plan"] + ["Tool: web_search (Tavily)", f"Web Results: {len(results)}"],
    }
