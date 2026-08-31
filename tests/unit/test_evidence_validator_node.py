from unittest.mock import MagicMock

from app.agents.nodes import evidence_validator as validator
from app.agents.nodes.evidence_validator import Coverage, EvidenceValidation


def _state(evidence=None, query="does metformin cut CV mortality in non-diabetics", **overrides):
    base = {
        "evidence": evidence or [],
        "messages": [{"role": "user", "content": query}],
        "current_query": query,
        "rerank_top_score": 0.8,
        "plan": ["Start"],
    }
    base.update(overrides)
    return base


def _verdict(sufficient, reason, cov=(True, True, True), notes="n"):
    return EvidenceValidation(
        sufficient=sufficient,
        reason=reason,
        coverage=Coverage(intervention=cov[0], population=cov[1], outcome=cov[2]),
        notes=notes,
    )


def test_sufficient_verdict_is_recorded(monkeypatch):
    monkeypatch.setattr(
        validator, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=_verdict(True, "sufficient"))),
    )
    ev = [{"pmid": "1", "claim": "c", "evidence": "e", "study_type": "RCT", "confidence": "high"}]

    result = validator.evidence_validator_node(_state(evidence=ev))

    assert result["evidence_verdict"]["sufficient"] is True
    assert result["evidence_verdict"]["reason"] == "sufficient"
    assert any("Verdict: sufficient" in note for note in result["plan"])


def test_population_mismatch_verdict_is_recorded(monkeypatch):
    monkeypatch.setattr(
        validator, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=_verdict(False, "population_mismatch", cov=(True, False, True)))),
    )

    result = validator.evidence_validator_node(_state())

    assert result["evidence_verdict"]["sufficient"] is False
    assert result["evidence_verdict"]["reason"] == "population_mismatch"
    assert result["evidence_verdict"]["coverage"]["population"] is False
    assert any("Verdict: population_mismatch" in note for note in result["plan"])


def test_corpus_failure_verdict_with_no_evidence(monkeypatch):
    captured = {}

    def _capture(prompt):
        captured["prompt"] = prompt
        return _verdict(False, "corpus_failure")

    monkeypatch.setattr(validator, "structured_llm", MagicMock(invoke=_capture))

    result = validator.evidence_validator_node(_state(evidence=[], rerank_top_score=0.0))

    assert result["evidence_verdict"]["reason"] == "corpus_failure"
    # rerank hint should note that no live search ran when there's no score
    assert "no live search ran" in captured["prompt"]
