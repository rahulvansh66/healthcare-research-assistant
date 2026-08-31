import uuid
from unittest.mock import MagicMock

from app.agents.graph import rag_agent
from app.agents.nodes import (
    claim_verifier,
    evidence_extractor,
    evidence_validator,
    planner,
    query_rewriter,
    responder,
    retriever,
)
from app.agents.nodes.claim_verifier import CritiqueResult
from app.agents.nodes.evidence_extractor import EvidenceExtraction, EvidenceRecord
from app.agents.nodes.evidence_validator import Coverage, EvidenceValidation
from app.agents.nodes.planner import PlannerDecision
from app.agents.nodes.query_rewriter import QueryRewrite


def _config():
    """A fresh thread per test — the real Postgres checkpointer persists state
    across runs, so reusing a fixed thread_id would leak turns between runs."""
    return {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}


def _completion(content):
    """A stand-in for the streaming Portkey response: an iterable of one delta
    chunk. ``extract_cache_status`` finds no ``_raw_response`` attr on a list and
    falls back to 'MISS'."""
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=content))]
    return [chunk]


def _initial_state(query: str) -> dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "current_query": query,
        "search_params": None,
        "pubmed_search_requested": False,
        "fresh_search_requested": False,
        "direct_pmid": None,
        "retrieval_source": "none",
        "retrieval_attempts": 0,
        "rerank_top_score": 0.0,
        "documents": [],
        "evidence": [],
        "evidence_verdict": None,
        "citations": [],
        "web_results": None,
        "used_web_fallback": False,
        "verify_attempts": 0,
        "unsupported_claims": [],
        "needs_regeneration": False,
        "_background_store_payload": None,
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }


def _sufficient():
    return EvidenceValidation(
        sufficient=True,
        reason="sufficient",
        coverage=Coverage(intervention=True, population=True, outcome=True),
        notes="covered",
    )


def _verdict(reason, cov=(False, False, False)):
    return EvidenceValidation(
        sufficient=False,
        reason=reason,
        coverage=Coverage(intervention=cov[0], population=cov[1], outcome=cov[2]),
        notes=reason,
    )


def _mock_clinical_retrieval(monkeypatch, doc):
    monkeypatch.setattr(
        planner, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=PlannerDecision(intent="CLINICAL", search_query="widget therapy"))),
    )
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": [doc["pmid"]], "total_results": 1})
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [doc])
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, texts, top_n: [(t, 0.9) for t in texts])
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)


def test_conversational_turn_skips_retrieval_and_evidence(monkeypatch):
    monkeypatch.setattr(
        planner, "structured_llm", MagicMock(invoke=MagicMock(return_value=PlannerDecision(intent="CONVERSATIONAL")))
    )
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("Hello! How can I help?")),
    )
    retriever_spy = MagicMock(side_effect=AssertionError("retriever should not run for conversational turns"))
    monkeypatch.setattr(retriever, "search_pubmed", retriever_spy)

    config = _config()
    result = rag_agent.invoke(_initial_state("hello"), config=config)

    assert result["final_answer"] == "Hello! How can I help?"
    assert result["citations"] == []
    assert result["retrieval_source"] == "none"


def test_clinical_turn_flows_through_pipeline_to_grounded_answer(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111", title="Widget Trial", abstract="Widget therapy reduced symptoms.")
    _mock_clinical_retrieval(monkeypatch, doc)

    record = EvidenceRecord(pmid="111", claim="Reduces symptoms", evidence="Widget therapy reduced symptoms.", study_type="RCT", confidence="high")
    monkeypatch.setattr(
        evidence_extractor, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=EvidenceExtraction(records=[record], insufficient_evidence=False))),
    )
    monkeypatch.setattr(
        evidence_validator, "structured_llm", MagicMock(invoke=MagicMock(return_value=_sufficient()))
    )
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("Widget therapy reduces symptoms [PMID 111].")),
    )
    monkeypatch.setattr(
        claim_verifier, "structured_critique_llm",
        MagicMock(invoke=MagicMock(return_value=CritiqueResult(all_claims_supported=True, unsupported_claims=[]))),
    )

    config = _config()
    result = rag_agent.invoke(_initial_state("does widget therapy work"), config=config)

    assert result["retrieval_source"] == "live_pubmed"
    assert result["evidence_verdict"]["sufficient"] is True
    assert result["final_answer"] == "Widget therapy reduces symptoms [PMID 111]."
    assert result["citations"] == [{
        "pmid": "111",
        "title": "Widget Trial",
        "journal": doc["journal"],
        "year": doc["year"],
        "study_type": "RCT",
        "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
    }]
    # exactly one assistant turn is checkpointed despite the responder -> claim_verifier hop
    assert [m["role"] for m in result["messages"]].count("assistant") == 1


def test_unsupported_claims_regenerate_then_caveat(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111", title="Widget Trial", abstract="Widget therapy reduced symptoms.")
    _mock_clinical_retrieval(monkeypatch, doc)

    record = EvidenceRecord(pmid="111", claim="Reduces symptoms", evidence="Widget therapy reduced symptoms.", study_type="RCT", confidence="high")
    monkeypatch.setattr(
        evidence_extractor, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=EvidenceExtraction(records=[record], insufficient_evidence=False))),
    )
    monkeypatch.setattr(
        evidence_validator, "structured_llm", MagicMock(invoke=MagicMock(return_value=_sufficient()))
    )
    create_mock = MagicMock(return_value=_completion("Widget therapy cures all disease permanently [PMID 111]."))
    monkeypatch.setattr(responder.portkey_client.chat.completions, "create", create_mock)
    monkeypatch.setattr(
        claim_verifier, "structured_critique_llm",
        MagicMock(invoke=MagicMock(return_value=CritiqueResult(
            all_claims_supported=False,
            unsupported_claims=["\"cures all disease permanently\" is not supported by the evidence"],
        ))),
    )

    config = _config()
    result = rag_agent.invoke(_initial_state("does widget therapy work"), config=config)

    # one regeneration (CLAIM_VERIFY_MAX_RETRIES=1) => responder invoked twice
    assert create_mock.call_count == 2
    assert "Self-Check Note" in result["final_answer"]
    assert any("Claim verification:" in note for note in result["plan"])
    assert [m["role"] for m in result["messages"]].count("assistant") == 1


def test_query_failure_triggers_rewrite_cycle_then_answers(monkeypatch, make_pubmed_document):
    hit = make_pubmed_document(pmid="222", title="Pediatric Widget Trial", abstract="Widget therapy helped children.")

    monkeypatch.setattr(
        planner, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=PlannerDecision(intent="CLINICAL", search_query="widget therapy"))),
    )
    search_calls = []

    def fake_search(query, **kwargs):
        search_calls.append(query)
        return {"pmids": ["222"], "total_results": 1}

    monkeypatch.setattr(retriever, "search_pubmed", fake_search)
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [hit])
    monkeypatch.setattr(retriever, "rerank_documents", lambda q, texts, top_n: [(t, 0.9) for t in texts])
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)

    record = EvidenceRecord(pmid="222", claim="Helps children", evidence="Widget therapy helped children.", study_type="RCT", confidence="high")
    monkeypatch.setattr(
        evidence_extractor, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=EvidenceExtraction(records=[record], insufficient_evidence=False))),
    )
    # first validation: query_failure -> rewrite; second: sufficient -> answer
    monkeypatch.setattr(
        evidence_validator, "structured_llm",
        MagicMock(invoke=MagicMock(side_effect=[_verdict("query_failure"), _sufficient()])),
    )
    monkeypatch.setattr(
        query_rewriter, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=QueryRewrite(rewritten_query="widget therapy pediatric"))),
    )
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("Widget therapy helps children [PMID 222].")),
    )
    monkeypatch.setattr(
        claim_verifier, "structured_critique_llm",
        MagicMock(invoke=MagicMock(return_value=CritiqueResult(all_claims_supported=True, unsupported_claims=[]))),
    )

    config = _config()
    result = rag_agent.invoke(_initial_state("does widget therapy work"), config=config)

    assert search_calls == ["widget therapy", "widget therapy pediatric"]
    assert result["retrieval_attempts"] == 1
    assert result["final_answer"] == "Widget therapy helps children [PMID 222]."


def test_corpus_failure_routes_to_web_fallback(monkeypatch):
    monkeypatch.setattr(
        planner, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=PlannerDecision(intent="CLINICAL", search_query="nonexistent condition xyz"))),
    )
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": [], "total_results": 0})
    monkeypatch.setattr(
        evidence_validator, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=_verdict("corpus_failure"))),
    )
    monkeypatch.setattr("app.agents.nodes.web_search.tavily_search", lambda q: [])
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("> ⚠️ This answer is from a general web search...\nNo reliable answer found.")),
    )

    config = _config()
    result = rag_agent.invoke(_initial_state("nonexistent condition xyz"), config=config)

    assert result["used_web_fallback"] is True
    assert result["evidence"] == []
    assert result["citations"] == []
    assert result["documents"] == []
