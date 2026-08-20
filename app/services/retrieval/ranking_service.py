import time

import logfire
import requests

from app.config import settings


def rerank_documents(query: str, documents: list[str], top_n: int = None) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically,
    via Jina's hosted reranker API.
    """
    if not documents:
        return []

    if top_n is None:
        top_n = settings.EVIDENCE_TOP_N

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to Jina Reranker...")

    try:
        response = requests.post(
            settings.JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
            timeout=settings.JINA_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json()["results"]

        reranked_docs = [documents[res["index"]] for res in results]

        duration = time.time() - start_time
        top_score = results[0]["relevance_score"] if results else "N/A"
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Top relevance score: {top_score}")

        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Jina reranking failed: {e}")
        # Fallback to the original order to ensure the user still gets an answer
        return documents[:top_n]
