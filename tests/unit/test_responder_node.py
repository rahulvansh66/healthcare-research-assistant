from unittest.mock import MagicMock

from app.agents.nodes import responder


def _completion(content, cache_status=None):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    if cache_status is not None:
        resp._raw_response = MagicMock(headers={"x-portkey-cache-status": cache_status})
    else:
        resp._raw_response = None
        resp._response = None
        resp._http_response = None
    return resp


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


def test_evidence_grounded_response_derives_citations_deterministically(monkeypatch, make_pubmed_document):
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

    try:
        responder.generate_node(_state("CONVERSATIONAL"))
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass
