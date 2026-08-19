import uuid
from datetime import datetime, timezone

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query, embed_texts, get_embedding_dim
from app.agents.state import PubMedDocument

# Session-scoped cache of live PubMed results — one shared collection, filtered
# by a thread_id payload field, not one collection per session (avoids an
# unbounded per-thread collection leak).

_NAMESPACE = uuid.UUID("6f1b1a2e-6b9f-4a3a-9c1a-8e2d7f5b0c11")

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
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
    Vector search over this thread's previously-stored PubMed results.
    Never raises: returns [] on failure.
    """
    limit = limit or settings.SESSION_CACHE_QUERY_LIMIT
    try:
        _ensure_collection()
        query_vector = embed_query(query)

        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="thread_id", match=models.MatchValue(value=thread_id))]
            ),
            limit=limit,
            with_payload=True,
        )

        results = []
        for res in response.points:
            payload = res.payload or {}
            results.append({
                "pmid": payload.get("pmid", ""),
                "title": payload.get("title", ""),
                "abstract": payload.get("abstract", ""),
                "journal": payload.get("journal", ""),
                "year": payload.get("year"),
                "pub_types": payload.get("pub_types", []),
                "authors": payload.get("authors", []),
                "score": res.score,
            })
        return results
    except Exception as e:
        logfire.error(f"❌ Session cache query failed: {e}")
        return []


def store_session_results(thread_id: str, query: str, documents: list[PubMedDocument]) -> None:
    """
    Embed and upsert PubMed documents into this thread's session cache.
    Meant to run as a background task after the response has already been
    sent — never raises out to the caller.
    """
    if not documents:
        return
    try:
        _ensure_collection()
        texts = [doc["abstract"] or doc["title"] for doc in documents]
        vectors = embed_texts(texts)
        stored_at = datetime.now(timezone.utc).isoformat()

        points = []
        for doc, vector in zip(documents, vectors):
            point_id = str(uuid.uuid5(_NAMESPACE, f"{thread_id}:{doc['pmid']}"))
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "thread_id": thread_id,
                    "source_query": query,
                    "stored_at": stored_at,
                    **doc,
                },
            ))

        client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
        logfire.info(f"Stored {len(points)} PubMed results in session cache for thread {thread_id}.")
    except Exception as e:
        logfire.error(f"❌ Failed to store session cache results: {e}")
