import logfire

from app.agents.state import WebResult
from app.config import settings

_client = None


def _get_client():
    """Lazy singleton TavilyClient. Returns None when no API key is configured
    so callers degrade gracefully instead of raising."""
    global _client
    if _client is not None:
        return _client
    if not settings.TAVILY_API_KEY:
        return None
    from tavily import TavilyClient

    _client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    return _client


def search(query: str) -> list[WebResult]:
    """
    Web-search fallback used only when the evidence validator returns
    ``corpus_failure``. Returns ``[]`` (and logs a warning) when the Tavily key
    is missing or the call fails — the web_search node treats an empty list as
    "no external sources found" and the responder emits the insufficient-evidence
    message rather than erroring.
    """
    client = _get_client()
    if client is None:
        logfire.warning("Tavily web-search fallback skipped: TAVILY_API_KEY not set.")
        return []

    try:
        with logfire.span("🌐 Tavily Web Search"):
            response = client.search(
                query=query,
                max_results=settings.TAVILY_MAX_RESULTS,
                search_depth=settings.TAVILY_SEARCH_DEPTH,
                timeout=settings.WEB_SEARCH_REQUEST_TIMEOUT,
            )
        results = response.get("results", []) if isinstance(response, dict) else []
        return [
            WebResult(
                title=(r.get("title") or r.get("url") or "Untitled"),
                url=r.get("url", ""),
                snippet=(r.get("content") or "").strip(),
            )
            for r in results
            if r.get("url")
        ]
    except Exception as e:
        logfire.error(f"❌ Tavily web search failed: {e}")
        return []
