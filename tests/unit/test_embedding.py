from unittest.mock import MagicMock, patch

from app.config import settings
from app.services.retrieval import embedding


def _mock_response(data):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": data}
    return resp


def test_get_embedding_dim_matches_config():
    assert embedding.get_embedding_dim() == settings.JINA_EMBEDDING_DIM


@patch("app.services.retrieval.embedding.requests.post")
def test_embed_query_returns_single_vector(mock_post):
    mock_post.return_value = _mock_response([{"index": 0, "embedding": [0.1, 0.2]}])

    vector = embedding.embed_query("what is widget therapy")

    assert vector == [0.1, 0.2]
    payload = mock_post.call_args.kwargs["json"]
    assert payload["task"] == "retrieval.query"
    assert payload["input"] == ["what is widget therapy"]


@patch("app.services.retrieval.embedding.requests.post")
def test_embed_query_reorders_by_index(mock_post):
    mock_post.return_value = _mock_response([
        {"index": 1, "embedding": [9.0]},
        {"index": 0, "embedding": [1.0]},
    ])

    vector = embedding.embed_query("q")

    assert vector == [1.0]


@patch("app.services.retrieval.embedding.requests.post")
def test_embed_texts_batches_requests(mock_post):
    batch_size = settings.JINA_EMBEDDING_BATCH_SIZE
    texts = [f"text-{i}" for i in range(batch_size + 5)]

    def fake_post(url, headers, json, timeout):
        n = len(json["input"])
        return _mock_response([{"index": i, "embedding": [float(i)]} for i in range(n)])

    mock_post.side_effect = fake_post

    vectors = embedding.embed_texts(texts)

    assert mock_post.call_count == 2
    assert len(vectors) == len(texts)


@patch("app.services.retrieval.embedding.requests.post")
def test_embed_texts_uses_passage_task(mock_post):
    mock_post.return_value = _mock_response([{"index": 0, "embedding": [1.0]}])

    embedding.embed_texts(["one text"])

    payload = mock_post.call_args.kwargs["json"]
    assert payload["task"] == "retrieval.passage"
