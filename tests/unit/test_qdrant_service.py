from unittest.mock import MagicMock

import pytest

from app.services.retrieval import qdrant_service


@pytest.fixture(autouse=True)
def reset_collection_ready(monkeypatch):
    monkeypatch.setattr(qdrant_service, "_collection_ready", True)


def _mock_point(pmid, score, title="T", chunk_text="chunk"):
    point = MagicMock()
    point.score = score
    point.payload = {
        "pmid": pmid,
        "title": title,
        "abstract": "abs",
        "journal": "J",
        "year": "2024",
        "pub_types": [],
        "authors": [],
        "has_full_text": False,
        "chunk_text": chunk_text,
    }
    return point


def test_query_session_cache_groups_chunks_by_pmid(monkeypatch):
    response = MagicMock()
    response.points = [
        _mock_point("111", score=0.9, chunk_text="best chunk"),
        _mock_point("111", score=0.4, chunk_text="worse chunk"),
        _mock_point("222", score=0.6, chunk_text="other doc"),
    ]
    mock_client = MagicMock()
    mock_client.query_points.return_value = response
    monkeypatch.setattr(qdrant_service, "client", mock_client)
    monkeypatch.setattr(qdrant_service, "embed_query", lambda q: [0.0, 0.0])

    results = qdrant_service.query_session_cache("widget therapy", thread_id="t1", limit=10)

    assert [r["pmid"] for r in results] == ["111", "222"]
    assert results[0]["score"] == 0.9
    assert results[0]["full_text_markdown"] == "best chunk\n\n---\n\nworse chunk"


def test_query_session_cache_respects_limit(monkeypatch):
    response = MagicMock()
    response.points = [_mock_point(str(i), score=1.0 - i / 10) for i in range(5)]
    mock_client = MagicMock()
    mock_client.query_points.return_value = response
    monkeypatch.setattr(qdrant_service, "client", mock_client)
    monkeypatch.setattr(qdrant_service, "embed_query", lambda q: [0.0])

    results = qdrant_service.query_session_cache("q", thread_id="t1", limit=2)

    assert len(results) == 2


def test_query_session_cache_returns_empty_on_failure(monkeypatch):
    mock_client = MagicMock()
    mock_client.query_points.side_effect = RuntimeError("qdrant is down")
    monkeypatch.setattr(qdrant_service, "client", mock_client)
    monkeypatch.setattr(qdrant_service, "embed_query", lambda q: [0.0])

    assert qdrant_service.query_session_cache("q", thread_id="t1") == []


def test_store_session_results_skips_empty_documents(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(qdrant_service, "client", mock_client)

    qdrant_service.store_session_results(thread_id="t1", query="q", documents=[])

    mock_client.upsert.assert_not_called()


def test_store_session_results_chunks_embeds_and_upserts(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111", title="Doc One", abstract="Some abstract content.")
    mock_client = MagicMock()
    monkeypatch.setattr(qdrant_service, "client", mock_client)
    monkeypatch.setattr(qdrant_service, "chunk_markdown", lambda md: [{"text": "chunk 1", "header_path": "Doc One"}])
    monkeypatch.setattr(qdrant_service, "embed_texts", lambda texts: [[0.1, 0.2] for _ in texts])

    qdrant_service.store_session_results(thread_id="t1", query="q", documents=[doc])

    mock_client.upsert.assert_called_once()
    points = mock_client.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["pmid"] == "111"
    assert points[0].payload["thread_id"] == "t1"
    assert points[0].payload["chunk_text"] == "chunk 1"


def test_store_session_results_never_raises_on_failure(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document()
    mock_client = MagicMock()
    monkeypatch.setattr(qdrant_service, "client", mock_client)
    monkeypatch.setattr(qdrant_service, "chunk_markdown", lambda md: (_ for _ in ()).throw(RuntimeError("boom")))

    qdrant_service.store_session_results(thread_id="t1", query="q", documents=[doc])
