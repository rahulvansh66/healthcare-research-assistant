import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- VECTOR DB (QDRANT) — session-scoped cache of live PubMed results,
    #     filtered by thread_id payload. Not a static pre-ingested corpus. ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "pubmed_session_cache"

    # --- PUBMED / NCBI E-UTILITIES ---
    # Secrets: .env only, never hardcoded.
    NCBI_API_KEY = os.getenv("NCBI_API_KEY")              # optional; unlocks 10 req/s vs 3 req/s
    NCBI_CONTACT_EMAIL = os.getenv("NCBI_CONTACT_EMAIL")  # required by NCBI usage guidelines
    NCBI_TOOL_NAME = os.getenv("NCBI_TOOL_NAME", "medico-healthcare-research-assistant")
    NCBI_EUTILS_BASE_URL = os.getenv("NCBI_EUTILS_BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
    NCBI_REQUEST_TIMEOUT = int(os.getenv("NCBI_REQUEST_TIMEOUT", "15"))

    # --- SESSION CACHE / RETRIEVAL TUNING ---
    PUBMED_FETCH_LIMIT = int(os.getenv("PUBMED_FETCH_LIMIT", "20"))
    PUBMED_STORE_TOP_N = int(os.getenv("PUBMED_STORE_TOP_N", "10"))
    EVIDENCE_TOP_N = int(os.getenv("EVIDENCE_TOP_N", "5"))
    SESSION_CACHE_QUERY_LIMIT = int(os.getenv("SESSION_CACHE_QUERY_LIMIT", "10"))
    SESSION_CACHE_RELEVANCE_THRESHOLD = float(os.getenv("SESSION_CACHE_RELEVANCE_THRESHOLD", "0.35"))

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "openai/gpt-oss-20b"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG = os.getenv("PORTKEY_CONFIG", "portkey_config")  # saved config id/slug (block_inline_config is on for this org)
    GROQ_SLUG =  "rv66-groq-healthcare-agent"          # primary virtual key: @rv66-groq-healthcare-agent/openai/gpt-oss-20b
    GROQ_SLUG_2 = "rv-dau-new-groq-healthcare-agent"   # fallback virtual key: @rv-dau-new-groq-healthcare-agent/openai/gpt-oss-20b

    
    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
