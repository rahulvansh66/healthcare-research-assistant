import logfire
import requests

from app.config import settings


# ── Local model fallback (unused — jina-embeddings-v3 is the only embedding path) ──
#
# def _load_main():
#     """Try loading the primary biomedical embedding model. Returns model or None."""
#     try:
#         from sentence_transformers import SentenceTransformer
#         model = SentenceTransformer("NeuML/biomedbert-small-embeddings")
#         model.encode(["probe"])
#         logfire.info("Loaded BiomedBERT-small-embeddings (22.7M params, 384-dim).")
#         return model
#     except Exception as e:
#         logfire.warning(f"Failed to load biomedbert-small-embeddings: {e}. Falling back to nano model.")
#         return None
#
#
# def _load_fallback():
#     from sentence_transformers import SentenceTransformer
#     logfire.info("Loading BiomedBERT-hash-nano-embeddings fallback (~1M params, 128-dim).")
#     return SentenceTransformer("NeuML/biomedbert-hash-nano-embeddings", trust_remote_code=True)


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model."""
    return settings.JINA_EMBEDDING_DIM


# ── Jina embeddings API ─────────────────────────────────────────────────────────

def _call_jina_embeddings(texts: list[str], task: str) -> list[list[float]]:
    response = requests.post(
        settings.JINA_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.JINA_EMBEDDINGS_MODEL,
            "task": task,
            "input": texts,
        },
        timeout=settings.JINA_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()["data"]
    data.sort(key=lambda item: item["index"])
    return [item["embedding"] for item in data]


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    return _call_jina_embeddings([query], task="retrieval.query")[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), settings.JINA_EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + settings.JINA_EMBEDDING_BATCH_SIZE]
        with logfire.span("Embed batch", model=settings.JINA_EMBEDDINGS_MODEL, start=i, size=len(batch)):
            all_embeddings.extend(_call_jina_embeddings(batch, task="retrieval.passage"))
    return all_embeddings
