import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables (secrets only — never read secrets from YAML)
load_dotenv()

_APP_ENV = os.getenv("APP_ENV", "dev")
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / f"{_APP_ENV}.yaml"

with open(_CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)


class Settings:
    # --- VECTOR DB (QDRANT) — session-scoped cache of live PubMed results,
    #     filtered by thread_id payload. Not a static pre-ingested corpus. ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")                # Qdrant cluster URL (secret: instance-specific)
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")                     # Qdrant auth key (secret)
    QDRANT_COLLECTION = _cfg["qdrant"]["collection"]                 # collection name for the per-thread PubMed cache
    QDRANT_REQUEST_TIMEOUT = int(_cfg["qdrant"]["request_timeout"])  # seconds — HTTP timeout for Qdrant requests

    # --- POSTGRES — durable LangGraph checkpoints + the sessions list (sidebar) ---
    DATABASE_URL = os.getenv("DATABASE_URL")  # secret: instance-specific (local dev: docker compose up -d postgres)

    # --- PUBMED / NCBI E-UTILITIES ---
    # Secrets: .env only, never hardcoded.
    NCBI_API_KEY = os.getenv("NCBI_API_KEY")              # optional; unlocks higher req/s rate limit
    NCBI_CONTACT_EMAIL = os.getenv("NCBI_CONTACT_EMAIL")  # required by NCBI usage guidelines
    NCBI_TOOL_NAME = _cfg["ncbi"]["tool_name"]                       # sent as the `tool` param on E-utilities requests
    NCBI_EUTILS_BASE_URL = _cfg["ncbi"]["eutils_base_url"]           # ESearch/EFetch base URL for live PubMed search
    NCBI_REQUEST_TIMEOUT = int(_cfg["ncbi"]["request_timeout"])      # seconds — HTTP timeout for NCBI requests
    NCBI_BIOC_BASE_URL = _cfg["ncbi"]["bioc_base_url"]               # BioC-PMC endpoint for open-access full-text fetch
    NCBI_RATE_LIMIT_WITH_KEY = float(_cfg["ncbi"]["rate_limit_with_key"])       # req/s throttle when NCBI_API_KEY is set
    NCBI_RATE_LIMIT_WITHOUT_KEY = float(_cfg["ncbi"]["rate_limit_without_key"]) # req/s throttle with no API key

    # --- SESSION CACHE / RETRIEVAL TUNING ---
    PUBMED_FETCH_LIMIT = int(_cfg["retrieval"]["pubmed_fetch_limit"])  # max articles fetched per live ESearch/EFetch call
    PUBMED_STORE_TOP_N = int(_cfg["retrieval"]["pubmed_store_top_n"])  # top-N reranked articles kept in the session cache after a live search
    EVIDENCE_TOP_N = int(_cfg["retrieval"]["evidence_top_n"])          # top-N reranked documents passed to evidence_agent for this turn
    SESSION_CACHE_QUERY_LIMIT = int(_cfg["retrieval"]["session_cache_query_limit"])  # max results fetched when querying the thread's session cache
    SESSION_CACHE_RELEVANCE_THRESHOLD = float(_cfg["retrieval"]["session_cache_relevance_threshold"])  # min top score to reuse cache instead of a fresh live search
    FULLTEXT_TOP_N = int(_cfg["retrieval"]["fulltext_top_n"])          # of the ranked docs, how many get open-access full text fetched vs. abstract-only
    FULLTEXT_EVIDENCE_CHAR_LIMIT = int(_cfg["retrieval"]["fulltext_evidence_char_limit"])  # max chars of full-text markdown per doc in the evidence prompt
    CHUNK_SIZE = int(_cfg["retrieval"]["chunk_size"])                  # characters per chunk when splitting full-text markdown for embedding
    CHUNK_OVERLAP = int(_cfg["retrieval"]["chunk_overlap"])            # character overlap between consecutive chunks
    SESSION_CACHE_CHUNK_FETCH_MULTIPLIER = int(_cfg["retrieval"]["session_cache_chunk_fetch_multiplier"])  # over-fetch factor (limit * this) for cached chunk queries
    RERANK_RELEVANCE_THRESHOLD = float(_cfg["retrieval"]["rerank_relevance_threshold"])  # min top Jina rerank score before CRAG rewrites the query and retries
    CRAG_MAX_RETRIES = int(_cfg["retrieval"]["crag_max_retries"])  # max query-rewrite + re-search attempts in retrieve_node

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")                    # primary Groq account key (secret)
    GROQ_MODEL = _cfg["groq"]["model"]                          # main reasoning model id used by planner/evidence/responder nodes
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")  # fallback Groq account key (secret)

    # --- JINA AI (embeddings + reranking) ---
    JINA_API_KEY = os.getenv("JINA_API_KEY")                            # Jina AI auth key (secret)
    JINA_EMBEDDINGS_URL = _cfg["jina"]["embeddings_url"]                # embeddings API endpoint
    JINA_EMBEDDINGS_MODEL = _cfg["jina"]["embeddings_model"]            # embedding model name sent to Jina
    JINA_EMBEDDING_DIM = int(_cfg["jina"]["embedding_dim"])             # vector dimension; must match the Qdrant collection's vector size
    JINA_EMBEDDING_BATCH_SIZE = int(_cfg["jina"]["embedding_batch_size"])  # texts embedded per Jina API call
    JINA_RERANK_URL = _cfg["jina"]["rerank_url"]                        # rerank API endpoint used to locally rerank retrieved PubMed results
    JINA_RERANK_MODEL = _cfg["jina"]["rerank_model"]                    # rerank model name sent to Jina
    JINA_REQUEST_TIMEOUT = int(_cfg["jina"]["request_timeout"])         # seconds — HTTP timeout for Jina requests

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")                                                  # Portkey auth key (secret)
    PORTKEY_CONFIG = os.getenv("PORTKEY_CONFIG", _cfg["portkey"]["config_id"])  # saved config id/slug (block_inline_config is on for this org)
    PORTKEY_GATEWAY_TEMPERATURE = float(_cfg["portkey"]["gateway_temperature"])  # sampling temperature for planner/evidence-agent structured-output calls
    PORTKEY_METADATA_ENVIRONMENT = _cfg["portkey"]["metadata_environment"]       # `environment` metadata tag on Portkey requests, for dashboard filtering
    PORTKEY_REQUEST_TIMEOUT = int(_cfg["portkey"]["request_timeout"])           # seconds — client-side HTTP timeout for LLM calls through the gateway
    GROQ_SLUG = _cfg["groq"]["primary_virtual_key"]    # primary virtual key: @{GROQ_SLUG}/openai/gpt-oss-20b
    GROQ_SLUG_2 = _cfg["groq"]["fallback_virtual_key"]  # fallback virtual key: @{GROQ_SLUG_2}/openai/gpt-oss-20b

    # --- GUARDRAILS ---
    GUARDRAILS_INTENT_MODEL = _cfg["guardrails"]["intent_model"]  # small/fast Groq model for the NeMo Guardrails intent-classification gate

    # --- RESPONDER ---
    RESPONDER_TEMPERATURE = float(_cfg["responder"]["temperature"])  # sampling temperature for the responder's final answer generation

    # --- INPUT VALIDATION ---
    MAX_QUERY_CHARS = int(_cfg["input_validation"]["max_query_chars"])            # max chars accepted for `q` in POST /query
    MAX_THREAD_ID_CHARS = int(_cfg["input_validation"]["max_thread_id_chars"])    # max chars accepted for thread_id
    MAX_HISTORY_CHARS = int(_cfg["input_validation"]["max_history_chars"])        # char budget for conversation history inlined into planner/responder prompts

    # --- RATE LIMITING ---
    RATE_LIMIT_QUERY = _cfg["rate_limit"]["query"]  # slowapi limit string for POST /query, keyed by client IP

    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", str(_cfg["langsmith"]["tracing_enabled"]).lower())  # enables LangSmith tracing for the agent
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")                            # LangSmith auth key (secret)
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", _cfg["langsmith"]["project"])    # LangSmith project traces are grouped under
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", _cfg["langsmith"]["endpoint"])  # LangSmith ingestion endpoint

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
