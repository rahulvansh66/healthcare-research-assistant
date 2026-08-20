# Test environment variables MUST be set before any `app.*` module is
# imported: several modules construct network clients (Portkey, ChatOpenAI,
# QdrantClient) at import time, and app.config reads .env via `load_dotenv()`
# with override=False, so values set here win over anything in a real .env.
import os

os.environ.setdefault("APP_ENV", "dev")
os.environ.setdefault("PORTKEY_API_KEY", "test-portkey-key")
os.environ.setdefault("PORTKEY_CONFIG", "test-portkey-config")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GROQ_FALLBACK_API_KEY", "test-groq-fallback-key")
os.environ.setdefault("NCBI_CONTACT_EMAIL", "test@example.com")
os.environ.setdefault("NCBI_API_KEY", "")
os.environ.setdefault("QDRANT_CLUSTER_ENDPOINT", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-qdrant-key")
os.environ.setdefault("JINA_API_KEY", "test-jina-key")
os.environ.setdefault("LOGFIRE_TOKEN", "")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGSMITH_API_KEY", "")

import pytest

from app.agents.state import PubMedDocument


@pytest.fixture
def sample_pubmed_document() -> PubMedDocument:
    return PubMedDocument(
        pmid="12345678",
        title="A Randomized Trial of Widget Therapy",
        abstract="Widget therapy reduced symptom severity versus placebo.",
        journal="Journal of Widget Medicine",
        year="2024",
        pub_types=["Randomized Controlled Trial"],
        authors=["Smith J", "Doe A"],
        has_full_text=False,
        full_text_markdown=None,
    )


@pytest.fixture
def make_pubmed_document():
    def _make(**overrides) -> PubMedDocument:
        base = PubMedDocument(
            pmid="00000000",
            title="Title",
            abstract="Abstract text.",
            journal="Journal",
            year="2023",
            pub_types=["Observational Study"],
            authors=["Author A"],
            has_full_text=False,
            full_text_markdown=None,
        )
        base.update(overrides)
        return base

    return _make


@pytest.fixture
def base_agent_state():
    """A minimal AgentState dict with every key the graph nodes expect."""
    return {
        "messages": [{"role": "user", "content": "What are the effects of widget therapy?"}],
        "current_query": "widget therapy effects",
        "search_params": None,
        "pubmed_search_requested": False,
        "fresh_search_requested": False,
        "retrieval_source": "none",
        "documents": [],
        "evidence": [],
        "citations": [],
        "_background_store_payload": None,
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "final_answer": "",
    }
