from unittest.mock import MagicMock

from app.gateway.client import extract_cache_status, get_langchain_llm
from app.config import settings


def test_get_langchain_llm_targets_portkey_gateway():
    llm = get_langchain_llm(feature="test-feature")

    assert llm.model_name == f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"


def test_extract_cache_status_returns_miss_when_no_raw_response_attrs():
    response = MagicMock(spec=[])

    assert extract_cache_status(response) == "MISS"


def test_extract_cache_status_reads_header_from_raw_response():
    response = MagicMock(spec=["_raw_response"])
    response._raw_response.headers = {"x-portkey-cache-status": "hit"}

    assert extract_cache_status(response) == "HIT"


def test_extract_cache_status_falls_back_to_miss_when_header_absent():
    response = MagicMock(spec=["_raw_response"])
    response._raw_response.headers = {}

    assert extract_cache_status(response) == "MISS"


def test_extract_cache_status_tries_multiple_attribute_paths():
    response = MagicMock(spec=["_raw_response", "_response", "_http_response"])
    response._raw_response = None
    response._response = None
    response._http_response.headers = {"x-portkey-cache-status": "miss"}

    assert extract_cache_status(response) == "MISS"
