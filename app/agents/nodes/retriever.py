import logfire
from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, PubMedDocument
from app.config import settings
from app.services.retrieval.pubmed_service import search_pubmed, get_pubmed_articles
from app.services.retrieval.qdrant_service import query_session_cache
from app.services.retrieval.ranking_service import rerank_documents


def _rerank_and_reattach(query: str, articles: list[PubMedDocument], top_n: int) -> list[PubMedDocument]:
    """
    ranking_service.rerank_documents() returns reranked text only, with no
    score/id mapping back to the source object. Rerank on "title\\nabstract"
    strings and re-attach the original PubMedDocument by matching text back
    positionally — avoids changing the reranker's public interface.
    """
    if not articles:
        return []

    texts = [f"{a['title']}\n{a['abstract']}" for a in articles]
    by_text = {}
    for text, article in zip(texts, articles):
        by_text.setdefault(text, article)

    reranked_texts = rerank_documents(query, texts, top_n=top_n)
    return [by_text[t] for t in reranked_texts if t in by_text]


def retrieve_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Decides whether to answer from this thread's session cache (Qdrant,
    filtered by thread_id) or to do a fresh live PubMed search, per:
      (a) this is the first message in the conversation, OR
      (b) the user forced a fresh search via the UI toggle, OR
      (c)/(d) the session cache is empty or has no sufficiently relevant hit, OR
      (e) the planner detected a "retry / search more" request
    -> live PubMed search. Otherwise, answer from the session cache.
    """
    thread_id = config["configurable"]["thread_id"]
    query = state["current_query"]
    params = state.get("search_params") or {}

    first_turn = len(state["messages"]) <= 1
    force_fresh = state.get("pubmed_search_requested", False)
    dissatisfied = state.get("fresh_search_requested", False)

    need_live_search = first_turn or force_fresh or dissatisfied
    documents: list[PubMedDocument] = []
    retrieval_source = "none"
    plan_notes = []

    with logfire.span("🔍 Retrieval"):
        if not need_live_search:
            cached = query_session_cache(query, thread_id, limit=settings.SESSION_CACHE_QUERY_LIMIT)
            if cached and max(c["score"] for c in cached) >= settings.SESSION_CACHE_RELEVANCE_THRESHOLD:
                documents = cached
                retrieval_source = "session_cache"
                plan_notes = ["Tool: session_cache", f"Cached Results: {len(documents)}"]
            else:
                need_live_search = True  # (c)/(d): empty or below relevance threshold

        background_payload = None
        if need_live_search:
            search_result = search_pubmed(
                query,
                max_results=settings.PUBMED_FETCH_LIMIT,
                date_from=params.get("date_from"),
                date_to=params.get("date_to"),
                publication_type=params.get("publication_type"),
            )
            pmids = search_result["pmids"]

            if not pmids:
                plan_notes = ["Tool: search_pubmed", "PMIDs Found: 0"]
            else:
                articles = get_pubmed_articles(pmids)
                with logfire.span("⚖️ Semantic Reranking"):
                    ranked = _rerank_and_reattach(query, articles, top_n=settings.PUBMED_STORE_TOP_N)

                documents = ranked[: settings.EVIDENCE_TOP_N]
                retrieval_source = "live_pubmed"
                background_payload = ranked
                plan_notes = [
                    "Tool: search_pubmed",
                    f"PMIDs Found: {len(pmids)}",
                    "Tool: get_pubmed_articles",
                    f"Reranked: top {len(documents)} of {len(ranked)} stored",
                ]

    status = "Found supporting literature." if documents else "No PubMed results found for this query."

    return {
        "documents": documents,
        "retrieval_source": retrieval_source,
        "_background_store_payload": background_payload,
        "status": status,
        "plan": state["plan"] + plan_notes,
    }
