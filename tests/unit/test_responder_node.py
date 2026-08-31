from unittest.mock import MagicMock

import pytest

from app.agents.nodes import responder


@pytest.fixture(autouse=True)
def _no_stream_writer(monkeypatch):
    """generate_node calls get_stream_writer() unconditionally; outside a live
    graph run that raises, so stub it with a no-op sink for direct-call tests."""
    monkeypatch.setattr(responder, "get_stream_writer", lambda: (lambda _delta: None))


class _Stream(list):
    """Iterable of delta chunks, with the optional cache-status header attr the
    native Portkey response would carry."""

    def __init__(self, chunks, cache_status=None):
        super().__init__(chunks)
        if cache_status is not None:
            self._raw_response = MagicMock(headers={"x-portkey-cache-status": cache_status})


def _completion(content, cache_status=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=content))]
    return _Stream([chunk], cache_status=cache_status)


def _state(query, evidence=None, documents=None, messages=None):
    return {
        "current_query": query,
        "messages": messages if messages is not None else [{"role": "user", "content": "hi"}],
        "evidence": evidence or [],
        "documents": documents or [],
        "plan": ["Start"],
    }


def test_conversational_response_uses_memory(monkeypatch):
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create", MagicMock(return_value=_completion("Hi there!"))
    )

    result = responder.generate_node(_state("CONVERSATIONAL"))

    assert result["final_answer"] == "Hi there!"
    assert result["citations"] == []
    assert result["messages"] == [{"role": "assistant", "content": "Hi there!"}]


def test_insufficient_evidence_response_has_no_citations(monkeypatch):
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("Not enough evidence.")),
    )

    result = responder.generate_node(_state("widget therapy", evidence=[]))

    assert result["citations"] == []
    assert result["final_answer"] == "Not enough evidence."
    # no evidence => terminal here, message appended by the responder
    assert result["messages"] == [{"role": "assistant", "content": "Not enough evidence."}]


def test_evidence_grounded_response_derives_citations_and_defers_message(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111", title="A Study", journal="J", year="2024")
    ev = {"pmid": "111", "claim": "c", "evidence": "e", "study_type": "RCT", "confidence": "high"}
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("Widget therapy is effective [PMID 111].")),
    )

    result = responder.generate_node(_state("widget therapy", evidence=[ev], documents=[doc]))

    assert result["citations"] == [{
        "pmid": "111",
        "title": "A Study",
        "journal": "J",
        "year": "2024",
        "study_type": "RCT",
        "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
    }]
    # grounded answers are appended to the transcript by claim_verifier, not here
    assert "messages" not in result


def test_dedupes_citations_by_pmid(make_pubmed_document, monkeypatch):
    doc = make_pubmed_document(pmid="111")
    evidence = [
        {"pmid": "111", "claim": "c1", "evidence": "e1", "study_type": "RCT", "confidence": "high"},
        {"pmid": "111", "claim": "c2", "evidence": "e2", "study_type": "RCT", "confidence": "moderate"},
    ]
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("answer")),
    )

    result = responder.generate_node(_state("widget therapy", evidence=evidence, documents=[doc]))

    assert len(result["citations"]) == 1


def test_web_fallback_response_has_no_citations(monkeypatch):
    state = _state("widget therapy", evidence=[])
    state["used_web_fallback"] = True
    state["web_results"] = [{"title": "Src", "url": "https://x.example", "snippet": "s"}]
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("From the web...")),
    )

    result = responder.generate_node(state)

    assert result["citations"] == []
    assert result["messages"] == [{"role": "assistant", "content": "From the web..."}]


def test_cache_hit_updates_status_and_plan(monkeypatch):
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("Hi there!", cache_status="HIT")),
    )

    result = responder.generate_node(_state("CONVERSATIONAL"))

    assert result["status"] == "Cache hit — instant response."
    assert result["plan"][-1] == "Cache: Hit ⚡"


def test_llm_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(side_effect=RuntimeError("gateway down")),
    )

    with pytest.raises(RuntimeError):
        responder.generate_node(_state("CONVERSATIONAL"))
