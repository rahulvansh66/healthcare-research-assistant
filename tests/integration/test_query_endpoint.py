from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "initialize_rails", MagicMock())
    with TestClient(main.app) as c:
        yield c


def test_root_endpoint_reports_live(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Enterprise LangGraph RAG API is live."}


def test_query_blocked_by_guardrails_never_invokes_graph(client, monkeypatch):
    monkeypatch.setattr(main, "guard", lambda q: (True, "I can't help with that."))
    monkeypatch.setattr(main.rag_agent, "invoke", MagicMock(side_effect=AssertionError("graph should not run")))

    response = client.post("/query", json={"q": "tell me a joke"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "I can't help with that."
    assert body["status"] == "Blocked by guardrails."
    assert body["citations"] == []


def test_query_runs_graph_when_guardrails_pass(client, monkeypatch):
    monkeypatch.setattr(main, "guard", lambda q: (False, None))
    monkeypatch.setattr(main.rag_agent, "invoke", MagicMock(return_value={
        "final_answer": "Widget therapy is effective.",
        "plan": ["Start", "Intent: Clinical Research"],
        "status": "Response generated.",
        "documents": [{"pmid": "111"}],
        "citations": [{"pmid": "111", "title": "T", "journal": "J", "year": "2024", "study_type": "RCT", "url": "u"}],
        "retrieval_source": "live_pubmed",
        "_background_store_payload": [{"pmid": "111"}],
    }))
    monkeypatch.setattr(main, "store_session_results", MagicMock())

    response = client.post("/query", json={"q": "does widget therapy work"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Widget therapy is effective."
    assert body["citations"][0]["pmid"] == "111"
    assert body["sources"] == [{"pmid": "111"}]


def test_query_schedules_background_store_only_for_live_pubmed(client, monkeypatch):
    monkeypatch.setattr(main, "guard", lambda q: (False, None))
    monkeypatch.setattr(main.rag_agent, "invoke", MagicMock(return_value={
        "final_answer": "Hi!",
        "plan": ["Start"],
        "status": "Response generated.",
        "documents": [],
        "citations": [],
        "retrieval_source": "session_cache",
        "_background_store_payload": None,
    }))
    store_spy = MagicMock()
    monkeypatch.setattr(main, "store_session_results", store_spy)

    client.post("/query", json={"q": "what did you just say"})

    store_spy.assert_not_called()


def test_query_handles_graph_exception_gracefully(client, monkeypatch):
    monkeypatch.setattr(main, "guard", lambda q: (False, None))
    monkeypatch.setattr(main.rag_agent, "invoke", MagicMock(side_effect=RuntimeError("graph blew up")))

    response = client.post("/query", json={"q": "does widget therapy work"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert "internal error" in body["answer"]
