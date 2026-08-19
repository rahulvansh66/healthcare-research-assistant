import logfire

BATCH_SIZE = 128
_MAIN_DIM = 384  # NeuML/biomedbert-small-embeddings (22.7M params)
_FALLBACK_DIM = 128  # NeuML/biomedbert-hash-nano-embeddings (~1M params)

_active_model = None
_model_type: str | None = None  # "main" or "fallback"


# ── Model initialisation ───────────────────────────────────────────────────────

def _load_main():
    """Try loading the primary biomedical embedding model. Returns model or None."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("NeuML/biomedbert-small-embeddings")
        model.encode(["probe"])
        logfire.info("Loaded BiomedBERT-small-embeddings (22.7M params, 384-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Failed to load biomedbert-small-embeddings: {e}. Falling back to nano model.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading BiomedBERT-hash-nano-embeddings fallback (~1M params, 128-dim).")
    return SentenceTransformer("NeuML/biomedbert-hash-nano-embeddings", trust_remote_code=True)


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    main = _load_main()
    if main:
        _active_model = main
        _model_type = "main"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _MAIN_DIM if _model_type == "main" else _FALLBACK_DIM


# ── Batch embedding ─────────────────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
