import uuid
from datetime import datetime, timezone

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.ingestion.chunking import chunk_markdown
from app.services.retrieval.embedding import embed_query, embed_texts, get_embedding_dim
from app.services.retrieval.markdown_builder import build_article_markdown
from app.agents.state import PubMedDocument

# Session-scoped cache of live PubMed results — one shared collection, filtered
# by a thread_id payload field, not one collection per session (avoids an
# unbounded per-thread collection leak).

_NAMESPACE = uuid.UUID("6f1b1a2e-6b9f-4a3a-9c1a-8e2d7f5b0c11")

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=settings.QDRANT_REQUEST_TIMEOUT
)

_collection_ready = False


def _ensure_collection():
    global _collection_ready
    if _collection_ready:
        return
    try:
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=get_embedding_dim(), distance=models.Distance.COSINE),
            )
            client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name="thread_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logfire.info(f"Created Qdrant collection '{settings.QDRANT_COLLECTION}'.")
        _collection_ready = True
    except Exception as e:
        logfire.error(f"❌ Failed to ensure Qdrant collection exists: {e}")


def query_session_cache(query: str, thread_id: str, limit: int = None) -> list[dict]:
    """
    Vector search over this thread's previously-stored PubMed chunks, then
    grouped back into one document per PMID (each carrying its
    highest-scoring chunks as `full_text_markdown`) so callers still see the
    same document shape as before chunking. Never raises: returns [] on
    failure.
    """
    limit = limit or settings.SESSION_CACHE_QUERY_LIMIT
    chunk_fetch_limit = limit * settings.SESSION_CACHE_CHUNK_FETCH_MULTIPLIER
    try:
        _ensure_collection()
        query_vector = embed_query(query)

        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="thread_id", match=models.MatchValue(value=thread_id))]
            ),
            limit=chunk_fetch_limit,
            with_payload=True,
        )

        by_pmid: dict[str, dict] = {}
        for res in response.points:
            payload = res.payload or {}
            pmid = payload.get("pmid", "")
            entry = by_pmid.setdefault(pmid, {
                "pmid": pmid,
                "title": payload.get("title", ""),
                "abstract": payload.get("abstract", ""),
                "journal": payload.get("journal", ""),
                "year": payload.get("year"),
                "pub_types": payload.get("pub_types", []),
                "authors": payload.get("authors", []),
                "has_full_text": payload.get("has_full_text", False),
                "score": res.score,
                "_chunks": [],
            })
            entry["_chunks"].append((res.score, payload.get("chunk_text", "")))
            entry["score"] = max(entry["score"], res.score)

        results = []
        for entry in by_pmid.values():
            chunk_texts = [t for _, t in sorted(entry.pop("_chunks"), key=lambda c: -c[0]) if t]
            entry["full_text_markdown"] = "\n\n---\n\n".join(chunk_texts) if chunk_texts else None
            results.append(entry)

        results.sort(key=lambda d: -d["score"])
        return results[:limit]
    except Exception as e:
        logfire.error(f"❌ Session cache query failed: {e}")
        return []


def store_session_results(thread_id: str, query: str, documents: list[PubMedDocument]) -> None:
    """
    Chunk each document's organized markdown (header-aware split, then
    size-bounded) and embed/upsert one Qdrant point per chunk into this
    thread's session cache. Meant to run as a background task after the
    response has already been sent — never raises out to the caller.
    """
    if not documents:
        return
    try:
        _ensure_collection()
        stored_at = datetime.now(timezone.utc).isoformat()

        doc_chunks: list[tuple[PubMedDocument, int, dict]] = []
        for doc in documents:
            markdown = build_article_markdown(doc)
            for chunk_index, chunk in enumerate(chunk_markdown(markdown)):
                doc_chunks.append((doc, chunk_index, chunk))

        if not doc_chunks:
            return

        vectors = embed_texts([chunk["text"] for _, _, chunk in doc_chunks])

        points = []
        for (doc, chunk_index, chunk), vector in zip(doc_chunks, vectors):
            point_id = str(uuid.uuid5(_NAMESPACE, f"{thread_id}:{doc['pmid']}:{chunk_index}"))
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "thread_id": thread_id,
                    "source_query": query,
                    "stored_at": stored_at,
                    "pmid": doc["pmid"],
                    "title": doc["title"],
                    "journal": doc["journal"],
                    "year": doc["year"],
                    "pub_types": doc["pub_types"],
                    "authors": doc["authors"],
                    "has_full_text": doc["has_full_text"],
                    "abstract": doc["abstract"],
                    "chunk_text": chunk["text"],
                    "header_path": chunk["header_path"],
                    "chunk_index": chunk_index,
                },
            ))

        client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
        logfire.info(
            f"Stored {len(points)} chunks ({len(documents)} PubMed results) "
            f"in session cache for thread {thread_id}."
        )
    except Exception as e:
        logfire.error(f"❌ Failed to store session cache results: {e}")
