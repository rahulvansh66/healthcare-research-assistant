import logfire
from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState, PubMedDocument
from app.config import settings
from app.services.retrieval.bioc_service import fetch_full_text_sections
from app.services.retrieval.markdown_builder import build_full_text_markdown
from app.services.retrieval.pubmed_service import search_pubmed, get_pubmed_articles
from app.services.retrieval.qdrant_service import query_session_cache
from app.services.retrieval.ranking_service import rerank_documents


def _attach_full_text(ranked: list[PubMedDocument]) -> None:
    """
    Fetch open-access full text (BioC-PMC) for the top FULLTEXT_TOP_N ranked
    articles, synchronously — this runs before the evidence extractor so full
    text (when available) is part of what it grounds answers in, not just
    what gets embedded into the session cache. Mutates documents in place.
    """
    for doc in ranked[: settings.FULLTEXT_TOP_N]:
        sections = fetch_full_text_sections(doc["pmid"])
        doc["has_full_text"] = sections is not None
        doc["full_text_markdown"] = build_full_text_markdown(sections) if sections else None
    for doc in ranked[settings.FULLTEXT_TOP_N :]:
        doc["has_full_text"] = False
        doc["full_text_markdown"] = None


def _rerank_and_reattach(query: str, articles: list[PubMedDocument], top_n: int) -> list[tuple[PubMedDocument, float]]:
    """
    Rerank on "title\\nabstract" strings and re-attach the original
    PubMedDocument to each (text, score) pair by matching text back
    positionally — avoids changing the reranker's public interface.
    """
    if not articles:
        return []

    texts = [f"{a['title']}\n{a['abstract']}" for a in articles]
    by_text = {}
    for text, article in zip(texts, articles):
        by_text.setdefault(text, article)

    reranked = rerank_documents(query, texts, top_n=top_n)
    return [(by_text[t], score) for t, score in reranked if t in by_text]


def retrieve_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    One retrieval pass. Decides whether to answer from this thread's session
    cache (Qdrant, filtered by thread_id) or to do a fresh live PubMed search,
    per:
      (a) this is the first message in the conversation, OR
      (b) the user forced a fresh search via the UI toggle, OR
      (c)/(d) the session cache is empty or has no sufficiently relevant hit, OR
      (e) the planner detected a "retry / search more" request, OR
      (f) we're on a retriever -> query_rewriter -> retriever cycle
          (``retrieval_attempts > 0``, ``current_query`` already rewritten).

    The CRAG query-rewrite retry is no longer an internal loop here — it's a real
    graph cycle owned by route_validator + the query_rewriter node. This node
    just runs the search with whatever ``current_query`` currently holds and
    reports the top rerank score via ``rerank_top_score`` for the validator.
    """
    thread_id = config["configurable"]["thread_id"]
    query = state["current_query"]
    params = state.get("search_params") or {}
    direct_pmid = state.get("direct_pmid")

    first_turn = len(state["messages"]) <= 1
    force_fresh = state.get("pubmed_search_requested", False)
    dissatisfied = state.get("fresh_search_requested", False)
    on_rewrite_cycle = state.get("retrieval_attempts", 0) > 0

    need_live_search = first_turn or force_fresh or dissatisfied or on_rewrite_cycle
    documents: list[PubMedDocument] = []
    retrieval_source = "none"
    rerank_top_score = 0.0
    plan_notes = []
    background_payload = None

    with logfire.span("🔍 Retrieval"):
        if direct_pmid:
            with logfire.span("📄 Direct PMID Fetch", pmid=direct_pmid):
                articles = get_pubmed_articles([direct_pmid])

            if not articles:
                plan_notes = ["Tool: get_pubmed_articles (direct PMID)", f"PMID {direct_pmid}: not found"]
            else:
                _attach_full_text(articles)
                documents = articles
                retrieval_source = "live_pubmed"
                background_payload = articles
                plan_notes = [
                    "Tool: get_pubmed_articles (direct PMID)",
                    f"PMID: {direct_pmid}",
                    f"Full Text: {'Yes' if articles[0]['has_full_text'] else 'No'}",
                ]

            status = "Found supporting literature." if documents else "No PubMed results found for this query."
            return {
                "documents": documents,
                "retrieval_source": retrieval_source,
                "rerank_top_score": 0.0,
                "_background_store_payload": background_payload,
                "status": status,
                "plan": state["plan"] + plan_notes,
            }

        if not need_live_search:
            cached = query_session_cache(query, thread_id, limit=settings.SESSION_CACHE_QUERY_LIMIT)
            if cached and max(c["score"] for c in cached) >= settings.SESSION_CACHE_RELEVANCE_THRESHOLD:
                documents = cached
                retrieval_source = "session_cache"
                plan_notes = ["Tool: session_cache", f"Cached Results: {len(documents)}"]
            else:
                need_live_search = True  # (c)/(d): empty or below relevance threshold

        if need_live_search:
            search_result = search_pubmed(
                query,
                max_results=settings.PUBMED_FETCH_LIMIT,
                date_from=params.get("date_from"),
                date_to=params.get("date_to"),
                publication_type=params.get("publication_type"),
            )
            pmids = search_result["pmids"]
            attempt_label = f"attempt {state.get('retrieval_attempts', 0) + 1}"

            if not pmids:
                plan_notes = ["Tool: search_pubmed", f"PMIDs Found: 0 ({attempt_label})"]
            else:
                articles = get_pubmed_articles(pmids)
                with logfire.span("⚖️ Semantic Reranking"):
                    ranked = _rerank_and_reattach(query, articles, top_n=settings.PUBMED_STORE_TOP_N)
                rerank_top_score = ranked[0][1] if ranked else 0.0
                ranked_docs = [d for d, _ in ranked]
                plan_notes = [
                    "Tool: search_pubmed",
                    f"PMIDs Found: {len(pmids)} ({attempt_label})",
                    f"Top Relevance: {rerank_top_score:.2f}",
                ]

                if ranked_docs:
                    with logfire.span("📄 Full Text Fetch"):
                        _attach_full_text(ranked_docs)
                    documents = ranked_docs[: settings.EVIDENCE_TOP_N]
                    retrieval_source = "live_pubmed"
                    background_payload = ranked_docs
                    full_text_count = sum(1 for d in documents if d["has_full_text"])
                    plan_notes.append(f"Reranked: top {len(documents)} of {len(ranked_docs)} stored")
                    plan_notes.append(f"Full Text: {full_text_count}/{len(documents)}")

    status = "Found supporting literature." if documents else "No PubMed results found for this query."

    return {
        "documents": documents,
        "retrieval_source": retrieval_source,
        "rerank_top_score": rerank_top_score,
        "_background_store_payload": background_payload,
        "status": status,
        "plan": state["plan"] + plan_notes,
    }
