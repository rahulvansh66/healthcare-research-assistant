from unittest.mock import MagicMock

from app.agents.graph import rag_agent
from app.agents.nodes import evidence, planner, responder, retriever
from app.agents.nodes.evidence import EvidenceExtraction, EvidenceRecord
from app.agents.nodes.planner import PlannerDecision
from app.agents.nodes.responder import CritiqueResult
from app.agents.nodes.retriever import QueryRewrite


def _completion(content):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp._raw_response = None
    resp._response = None
    resp._http_response = None
    return resp


def _initial_state(query: str) -> dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "current_query": query,
        "search_params": None,
        "pubmed_search_requested": False,
        "fresh_search_requested": False,
        "retrieval_source": "none",
        "documents": [],
        "evidence": [],
        "citations": [],
        "_background_store_payload": None,
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }


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

    config = {"configurable": {"thread_id": "test-thread-conversational"}}
    result = rag_agent.invoke(_initial_state("hello"), config=config)

    assert result["final_answer"] == "Hello! How can I help?"
    assert result["citations"] == []
    assert result["retrieval_source"] == "none"


def test_clinical_turn_flows_through_retrieval_and_evidence_to_grounded_answer(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111", title="Widget Trial", abstract="Widget therapy reduced symptoms.")

    monkeypatch.setattr(
        planner, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=PlannerDecision(intent="CLINICAL", search_query="widget therapy"))),
    )
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": ["111"], "total_results": 1})
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [doc])
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, texts, top_n: [(t, 0.9) for t in texts])
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)

    record = EvidenceRecord(pmid="111", claim="Reduces symptoms", evidence="Widget therapy reduced symptoms.", study_type="RCT", confidence="high")
    monkeypatch.setattr(
        evidence, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=EvidenceExtraction(records=[record], insufficient_evidence=False))),
    )
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("Widget therapy reduces symptoms [PMID 111].")),
    )
    monkeypatch.setattr(
        responder, "structured_critique_llm",
        MagicMock(invoke=MagicMock(return_value=CritiqueResult(all_claims_supported=True, unsupported_claims=[]))),
    )

    config = {"configurable": {"thread_id": "test-thread-clinical"}}
    result = rag_agent.invoke(_initial_state("does widget therapy work"), config=config)

    assert result["retrieval_source"] == "live_pubmed"
    assert result["final_answer"] == "Widget therapy reduces symptoms [PMID 111]."
    assert result["citations"] == [{
        "pmid": "111",
        "title": "Widget Trial",
        "journal": doc["journal"],
        "year": doc["year"],
        "study_type": "RCT",
        "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
    }]


def test_self_critique_appends_caveat_for_unsupported_claims(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111", title="Widget Trial", abstract="Widget therapy reduced symptoms.")

    monkeypatch.setattr(
        planner, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=PlannerDecision(intent="CLINICAL", search_query="widget therapy"))),
    )
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": ["111"], "total_results": 1})
    monkeypatch.setattr(retriever, "get_pubmed_articles", lambda pmids: [doc])
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, texts, top_n: [(t, 0.9) for t in texts])
    monkeypatch.setattr(retriever, "fetch_full_text_sections", lambda pmid: None)

    record = EvidenceRecord(pmid="111", claim="Reduces symptoms", evidence="Widget therapy reduced symptoms.", study_type="RCT", confidence="high")
    monkeypatch.setattr(
        evidence, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=EvidenceExtraction(records=[record], insufficient_evidence=False))),
    )
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("Widget therapy cures all disease permanently [PMID 111].")),
    )
    monkeypatch.setattr(
        responder, "structured_critique_llm",
        MagicMock(invoke=MagicMock(return_value=CritiqueResult(
            all_claims_supported=False,
            unsupported_claims=["\"cures all disease permanently\" is not supported by the evidence"],
        ))),
    )

    config = {"configurable": {"thread_id": "test-thread-critique"}}
    result = rag_agent.invoke(_initial_state("does widget therapy work"), config=config)

    assert "Self-Check Note" in result["final_answer"]
    assert "cures all disease permanently" in result["final_answer"]
    assert any("Self-critique: flagged" in note for note in result["plan"])


def test_clinical_turn_with_no_search_results_yields_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(
        planner, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=PlannerDecision(intent="CLINICAL", search_query="nonexistent condition xyz"))),
    )
    monkeypatch.setattr(retriever, "search_pubmed", lambda *a, **k: {"pmids": [], "total_results": 0})
    monkeypatch.setattr(
        retriever, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=QueryRewrite(rewritten_query="still nonexistent condition xyz"))),
    )
    monkeypatch.setattr(
        responder.portkey_client.chat.completions, "create",
        MagicMock(return_value=_completion("I could not find sufficient evidence for this question.")),
    )

    config = {"configurable": {"thread_id": "test-thread-no-results"}}
    result = rag_agent.invoke(_initial_state("nonexistent condition xyz"), config=config)

    assert result["evidence"] == []
    assert result["citations"] == []
    assert result["documents"] == []
