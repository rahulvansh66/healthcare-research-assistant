from unittest.mock import MagicMock, patch

import requests

from app.services.retrieval import ranking_service


def _mock_response(json_data):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_data
    return resp


def test_empty_documents_returns_empty_without_request():
    assert ranking_service.rerank_documents("query", []) == []


@patch("app.services.retrieval.ranking_service.requests.post")
def test_reorders_documents_by_reranker_response(mock_post):
    docs = ["doc A", "doc B", "doc C"]
    mock_post.return_value = _mock_response({
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
        ]
    })

    result = ranking_service.rerank_documents("query", docs, top_n=2)

    assert result == ["doc C", "doc A"]


@patch("app.services.retrieval.ranking_service.requests.post")
def test_sends_top_n_and_query_in_payload(mock_post):
    mock_post.return_value = _mock_response({"results": []})

    ranking_service.rerank_documents("my query", ["a", "b"], top_n=1)

    payload = mock_post.call_args.kwargs["json"]
    assert payload["query"] == "my query"
    assert payload["documents"] == ["a", "b"]
    assert payload["top_n"] == 1


@patch("app.services.retrieval.ranking_service.requests.post")
def test_falls_back_to_original_order_on_request_failure(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("down")

    result = ranking_service.rerank_documents("query", ["a", "b", "c"], top_n=2)

    assert result == ["a", "b"]


@patch("app.services.retrieval.ranking_service.requests.post")
def test_falls_back_to_original_order_on_http_error(mock_post):
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    mock_post.return_value = resp

    result = ranking_service.rerank_documents("query", ["a", "b", "c"])

    assert result == ["a", "b", "c"]
