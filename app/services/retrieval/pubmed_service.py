import threading
import time
import xml.etree.ElementTree as ET

import logfire
import requests

from app.config import settings
from app.agents.state import PubMedDocument

_ESEARCH_URL = f"{settings.NCBI_EUTILS_BASE_URL}/esearch.fcgi"
_EFETCH_URL = f"{settings.NCBI_EUTILS_BASE_URL}/efetch.fcgi"

_last_call_ts = 0.0
_lock = threading.Lock()


def _throttle():
    """NCBI rate limits: 3 req/s without an API key, 10 req/s with one."""
    min_interval = 1.0 / 10 if settings.NCBI_API_KEY else 1.0 / 3
    global _last_call_ts
    with _lock:
        elapsed = time.monotonic() - _last_call_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_call_ts = time.monotonic()


def _base_params() -> dict:
    params = {"tool": settings.NCBI_TOOL_NAME}
    if settings.NCBI_CONTACT_EMAIL:
        params["email"] = settings.NCBI_CONTACT_EMAIL
    if settings.NCBI_API_KEY:
        params["api_key"] = settings.NCBI_API_KEY
    return params


def _get(url: str, params: dict) -> requests.Response | None:
    """GET with throttling, a bounded retry, and degrade-to-None on failure."""
    for attempt in range(2):
        _throttle()
        try:
            response = requests.get(url, params=params, timeout=settings.NCBI_REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            logfire.warning(f"NCBI request failed (attempt {attempt + 1}/2): {e}")
    return None


def _build_esearch_query(query: str, date_from: str | None, date_to: str | None, publication_type: str | None) -> str:
    parts = [f'("{query}")']
    if publication_type:
        parts.append(f'("{publication_type}"[Publication Type])')
    if date_from or date_to:
        parts.append(f'("{date_from or "1900/01/01"}"[Date - Publication] : "{date_to or "3000/01/01"}"[Date - Publication])')
    return " AND ".join(parts)


def search_pubmed(
    query: str,
    max_results: int = None,
    date_from: str | None = None,
    date_to: str | None = None,
    publication_type: str | None = None,
) -> dict:
    """
    ESearch — find PMIDs matching a boolean query.
    Never raises: returns {"pmids": [], "total_results": 0} on any failure.
    """
    max_results = max_results or settings.PUBMED_FETCH_LIMIT
    term = _build_esearch_query(query, date_from, date_to, publication_type)

    params = {
        **_base_params(),
        "db": "pubmed",
        "retmode": "json",
        "retmax": max_results,
        "term": term,
    }

    with logfire.span("ESearch", query=term):
        response = _get(_ESEARCH_URL, params)
        if response is None:
            logfire.error("ESearch failed after retries — returning no results.")
            return {"pmids": [], "total_results": 0}

        try:
            data = response.json()
            result = data.get("esearchresult", {})
            pmids = result.get("idlist", [])
            total = int(result.get("count", 0))
            logfire.info(f"ESearch returned {len(pmids)} PMIDs (total_results={total})")
            return {"pmids": pmids, "total_results": total}
        except (ValueError, KeyError) as e:
            logfire.error(f"ESearch response parsing failed: {e}")
            return {"pmids": [], "total_results": 0}


def _text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def _parse_article(article_elem: ET.Element) -> PubMedDocument | None:
    try:
        pmid = _text(article_elem.find(".//PMID"))
        if not pmid:
            return None

        title = _text(article_elem.find(".//ArticleTitle"))

        abstract_parts = [_text(t) for t in article_elem.findall(".//AbstractText")]
        abstract = "\n".join(p for p in abstract_parts if p)

        journal = _text(article_elem.find(".//Journal/Title"))

        year = _text(article_elem.find(".//Journal/JournalIssue/PubDate/Year"))
        if not year:
            medline_date = _text(article_elem.find(".//Journal/JournalIssue/PubDate/MedlineDate"))
            year = medline_date[:4] if medline_date else None

        pub_types = [_text(pt) for pt in article_elem.findall(".//PublicationTypeList/PublicationType")]
        pub_types = [pt for pt in pub_types if pt]

        authors = []
        for author_elem in article_elem.findall(".//AuthorList/Author"):
            last_name = _text(author_elem.find("LastName"))
            initials = _text(author_elem.find("Initials"))
            if last_name:
                authors.append(f"{last_name} {initials}".strip())

        return PubMedDocument(
            pmid=pmid,
            title=title,
            abstract=abstract,
            journal=journal,
            year=year or None,
            pub_types=pub_types,
            authors=authors,
        )
    except Exception as e:
        logfire.warning(f"Failed to parse a PubMed article record: {e}")
        return None


def get_pubmed_articles(pmids: list[str]) -> list[PubMedDocument]:
    """
    EFetch — retrieve full records (title, abstract, journal, year, publication
    types, authors) for a batch of PMIDs in a single request, per NCBI's own
    batching guidance. Never raises: returns [] on total failure, skips
    individual malformed records rather than failing the whole batch.
    """
    if not pmids:
        return []

    params = {
        **_base_params(),
        "db": "pubmed",
        "rettype": "abstract",
        "retmode": "xml",
        "id": ",".join(pmids),
    }

    with logfire.span("EFetch", pmid_count=len(pmids)):
        response = _get(_EFETCH_URL, params)
        if response is None:
            logfire.error("EFetch failed after retries — returning no articles.")
            return []

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            logfire.error(f"EFetch XML parsing failed: {e}")
            return []

        articles = []
        for article_elem in root.findall(".//PubmedArticle"):
            doc = _parse_article(article_elem)
            if doc is not None:
                articles.append(doc)

        logfire.info(f"EFetch parsed {len(articles)}/{len(pmids)} article records")
        return articles
