import logfire
from pydantic import BaseModel

from app.agents.prompts.retriever import build_query_rewrite_prompt
from app.agents.state import AgentState
from app.gateway import get_langchain_llm

llm = get_langchain_llm(feature="query_rewriter")


class QueryRewrite(BaseModel):
    rewritten_query: str


structured_llm = llm.with_structured_output(QueryRewrite)


def query_rewriter_node(state: AgentState) -> dict:
    """
    One arm of the retriever <-> query_rewriter graph cycle. Reached when the
    evidence validator returned ``query_failure`` / ``population_mismatch`` and
    the retry budget (``CRAG_MAX_RETRIES``) isn't spent. Rewrites
    ``current_query`` with a small LLM call so the next retriever pass searches
    PubMed with better terms, and bumps ``retrieval_attempts``.
    """
    original_question = state["messages"][-1]["content"] if state["messages"] else state["current_query"]
    failed_query = state["current_query"]
    attempt = state.get("retrieval_attempts", 0) + 1
    verdict = state.get("evidence_verdict") or {}
    feedback = verdict.get("notes", "") if verdict.get("reason") != "sufficient" else ""

    with logfire.span("✏️ CRAG Query Rewrite", attempt=attempt):
        rewrite = structured_llm.invoke(
            build_query_rewrite_prompt(original_question, failed_query, attempt, feedback)
        )

    new_query = rewrite.rewritten_query.strip() or failed_query
    logfire.info(f"Query rewritten (attempt {attempt}): {new_query!r}")

    return {
        "current_query": new_query,
        "retrieval_attempts": attempt,
        "status": f"Rewriting search query (attempt {attempt}): {new_query}",
        "plan": state["plan"]
        + [
            "Tool: query_rewriter",
            f'CRAG attempt {attempt}: "{failed_query}" -> "{new_query}"',
        ],
    }
