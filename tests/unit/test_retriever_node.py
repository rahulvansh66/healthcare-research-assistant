from app.agents.nodes import retriever


def _config(thread_id="thread-1"):
    return {"configurable": {"thread_id": thread_id}}


def _state(query="widget therapy", messages=None, **overrides):
    base = {
        "messages": messages if messages is not None else [{"role": "user", "content": query}],
        "current_query": query,
        "search_params": None,
        "pubmed_search_requested": False,
        "fresh_search_requested": False,
        "direct_pmid": None,
        "plan": ["Start"],
    }
    base.update(overrides)
    return base


def test_first_turn_forces_live_search_and_reranks(monkeypatch, make_pubmed_document):
    articles = [
        make_pubmed_document(pmid="1", title="A", abstract="a"),
        make_pubmed_document(pmid="2", title="B", abstract="b"),
    ]
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": ["1", "2"], "total_results": 2})
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: articles)
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, texts, top_n: [(t, 0.9) for t in texts])
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)
    monkeypatch.setattr(retriever, "query_session_cache", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache should not be queried on first turn")))

    result = retriever.retrieve_node(_state(), _config())

    assert result["retrieval_source"] == "live_pubmed"
    assert [d["pmid"] for d in result["documents"]] == ["1", "2"]
    assert result["_background_store_payload"] is not None
    assert all(d["has_full_text"] is False for d in result["documents"])


def test_live_search_with_no_pmids_returns_no_documents(monkeypatch):
    # The CRAG rewrite retry is no longer an internal loop here — it's a graph
    # cycle owned by route_validator + the query_rewriter node. The retriever
    # just does one pass and reports the outcome.
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": [], "total_results": 0})

    result = retriever.retrieve_node(_state(), _config())

    assert result["documents"] == []
    assert result["retrieval_source"] == "none"
    assert result["rerank_top_score"] == 0.0
    assert any("PMIDs Found: 0" in note for note in result["plan"])


def test_low_relevance_still_returns_docs_and_reports_top_score(monkeypatch, make_pubmed_document):
    # Low rerank relevance is no longer retried in-node; the docs come back and
    # rerank_top_score is surfaced so the evidence validator can decide.
    doc = make_pubmed_document(pmid="1", title="Low", abstract="low relevance")

    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": ["1"], "total_results": 1})
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [doc])
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, texts, top_n: [(t, 0.1) for t in texts])
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)

    result = retriever.retrieve_node(_state(), _config())

    assert [d["pmid"] for d in result["documents"]] == ["1"]
    assert result["retrieval_source"] == "live_pubmed"
    assert result["rerank_top_score"] == 0.1
    assert any("Top Relevance: 0.10" in note for note in result["plan"])


def test_rewrite_cycle_forces_fresh_search_with_rewritten_query(monkeypatch, make_pubmed_document):
    # On a retriever -> query_rewriter -> retriever cycle, retrieval_attempts > 0
    # and current_query has already been rewritten; the retriever must re-run a
    # live search (never fall back to the session cache).
    history = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "widget therapy"},
    ]
    doc = make_pubmed_document(pmid="7", title="Rewritten hit", abstract="a")
    search_calls = []

    def fake_search(query, **kwargs):
        search_calls.append(query)
        return {"pmids": ["7"], "total_results": 1}

    monkeypatch.setattr(retriever, "search_pubmed", fake_search)
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [doc])
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, texts, top_n: [(t, 0.9) for t in texts])
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)
    monkeypatch.setattr(
        retriever, "query_session_cache",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache must be skipped on a rewrite cycle")),
    )

    result = retriever.retrieve_node(
        _state(query="widget therapy remission", messages=history, retrieval_attempts=1), _config()
    )

    assert search_calls == ["widget therapy remission"]
    assert result["retrieval_source"] == "live_pubmed"
    assert [d["pmid"] for d in result["documents"]] == ["7"]


def test_uses_session_cache_when_relevant_and_not_first_turn(monkeypatch, make_pubmed_document):
    history = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "widget therapy"},
    ]
    cached_doc = make_pubmed_document(pmid="5")
    cached_doc["score"] = 0.9
    monkeypatch.setattr(retriever, "query_session_cache", lambda *a, **k: [cached_doc])
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not search live")))

    result = retriever.retrieve_node(_state(messages=history), _config())

    assert result["retrieval_source"] == "session_cache"
    assert result["documents"] == [cached_doc]


def test_falls_through_to_live_search_when_cache_below_threshold(monkeypatch, make_pubmed_document):
    history = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "widget therapy"},
    ]
    low_score_doc = make_pubmed_document(pmid="5")
    low_score_doc["score"] = 0.01
    live_doc = make_pubmed_document(pmid="9", title="Live", abstract="live abstract")

    monkeypatch.setattr(retriever, "query_session_cache", lambda *a, **k: [low_score_doc])
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": ["9"], "total_results": 1})
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [live_doc])
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, texts, top_n: [(t, 0.9) for t in texts])
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)

    result = retriever.retrieve_node(_state(messages=history), _config())

    assert result["retrieval_source"] == "live_pubmed"
    assert [d["pmid"] for d in result["documents"]] == ["9"]


def test_fresh_search_requested_forces_live_search_even_with_cache_hit(monkeypatch, make_pubmed_document):
    history = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "search again"},
    ]
    live_doc = make_pubmed_document(pmid="9", title="Live", abstract="live abstract")
    monkeypatch.setattr(retriever, "query_session_cache", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache should be skipped")))
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": ["9"], "total_results": 1})
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [live_doc])
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, texts, top_n: [(t, 0.9) for t in texts])
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)

    result = retriever.retrieve_node(_state(messages=history, fresh_search_requested=True), _config())

    assert result["retrieval_source"] == "live_pubmed"


def test_attaches_full_text_only_up_to_fulltext_top_n(monkeypatch, make_pubmed_document):
    from app.config import settings

    articles = [make_pubmed_document(pmid=str(i), title=f"T{i}", abstract=f"a{i}") for i in range(settings.FULLTEXT_TOP_N + 2)]
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": [d["pmid"] for d in articles], "total_results": len(articles)})
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: articles)
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, texts, top_n: [(t, 0.9) for t in texts[:top_n]])
    fetched = []

    def fake_fetch(pmid):
        fetched.append(pmid)
        return [{"section": "INTRO", "text": "full text body"}]

    monkeypatch.setattr(retriever, "fetch_full_text_sections", fake_fetch)

    retriever.retrieve_node(_state(), _config())

    assert len(fetched) == settings.FULLTEXT_TOP_N


def test_direct_pmid_skips_search_and_rerank_and_cache(monkeypatch, make_pubmed_document):
    fetched_doc = make_pubmed_document(pmid="39796530", title="Direct", abstract="direct abstract")

    def fake_get_articles(pmids):
        assert pmids == ["39796530"]
        return [fetched_doc]

    monkeypatch.setattr(retriever, "get_pubmed_articles", fake_get_articles)
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not search live")))
    monkeypatch.setattr(retriever, "rerank_documents", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not rerank")))
    monkeypatch.setattr(retriever, "query_session_cache", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache should not be queried")))

    result = retriever.retrieve_node(_state(direct_pmid="39796530"), _config())

    assert result["retrieval_source"] == "live_pubmed"
    assert [d["pmid"] for d in result["documents"]] == ["39796530"]
    assert result["_background_store_payload"] == [fetched_doc]


def test_direct_pmid_not_found_returns_no_documents(monkeypatch):
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [])

    result = retriever.retrieve_node(_state(direct_pmid="00000000"), _config())

    assert result["documents"] == []
    assert result["retrieval_source"] == "none"
    assert any("not found" in note for note in result["plan"])


def test_direct_pmid_full_text_attached(monkeypatch, make_pubmed_document):
    fetched_doc = make_pubmed_document(pmid="39796530", title="Direct", abstract="direct abstract")
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [fetched_doc])

    fetched = []

    def fake_fetch(pmid):
        fetched.append(pmid)
        return [{"section": "INTRO", "text": "full text body"}]

    monkeypatch.setattr(retriever, "fetch_full_text_sections", fake_fetch)

    result = retriever.retrieve_node(_state(direct_pmid="39796530"), _config())

    assert fetched == ["39796530"]
    assert result["documents"][0]["has_full_text"] is True
