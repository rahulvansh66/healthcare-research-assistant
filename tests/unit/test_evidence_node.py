from unittest.mock import MagicMock

from app.agents.nodes import evidence
from app.agents.nodes.evidence import EvidenceExtraction, EvidenceRecord


def _state(documents, query="does widget therapy work"):
    return {
        "documents": documents,
        "messages": [{"role": "user", "content": query}],
        "current_query": query,
        "plan": ["Start"],
    }


def test_no_documents_short_circuits_without_calling_llm(monkeypatch, make_pubmed_document):
    mock_invoke = MagicMock()
    monkeypatch.setattr(evidence, "structured_llm", MagicMock(invoke=mock_invoke))

    result = evidence.evidence_agent_node(_state([]))

    assert result["evidence"] == []
    assert "Insufficient (no documents)" in result["plan"][-1]
    mock_invoke.assert_not_called()


def test_extracts_valid_evidence_records(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111", title="A Study", abstract="Widget therapy works.")
    record = EvidenceRecord(pmid="111", claim="Works", evidence="Widget therapy works.", study_type="RCT", confidence="high")
    extraction = EvidenceExtraction(records=[record], insufficient_evidence=False)
    monkeypatch.setattr(evidence, "structured_llm", MagicMock(invoke=MagicMock(return_value=extraction)))

    result = evidence.evidence_agent_node(_state([doc]))

    assert result["evidence"] == [record.model_dump()]
    assert "Evidence Records: 1" in result["plan"][-1]


def test_drops_records_citing_pmid_outside_retrieved_set(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111")
    valid_record = EvidenceRecord(pmid="111", claim="Works", evidence="e", study_type="RCT", confidence="high")
    hallucinated_record = EvidenceRecord(pmid="999", claim="Invented", evidence="e", study_type="RCT", confidence="high")
    extraction = EvidenceExtraction(records=[valid_record, hallucinated_record], insufficient_evidence=False)
    monkeypatch.setattr(evidence, "structured_llm", MagicMock(invoke=MagicMock(return_value=extraction)))

    result = evidence.evidence_agent_node(_state([doc]))

    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["pmid"] == "111"


def test_insufficient_evidence_flag_returns_empty(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111")
    extraction = EvidenceExtraction(records=[], insufficient_evidence=True)
    monkeypatch.setattr(evidence, "structured_llm", MagicMock(invoke=MagicMock(return_value=extraction)))

    result = evidence.evidence_agent_node(_state([doc]))

    assert result["evidence"] == []
    assert result["status"] == "Retrieved literature does not sufficiently answer the question."


def test_all_records_dropped_treated_as_insufficient(monkeypatch, make_pubmed_document):
    doc = make_pubmed_document(pmid="111")
    hallucinated_record = EvidenceRecord(pmid="999", claim="Invented", evidence="e", study_type="RCT", confidence="high")
    extraction = EvidenceExtraction(records=[hallucinated_record], insufficient_evidence=False)
    monkeypatch.setattr(evidence, "structured_llm", MagicMock(invoke=MagicMock(return_value=extraction)))

    result = evidence.evidence_agent_node(_state([doc]))

    assert result["evidence"] == []
