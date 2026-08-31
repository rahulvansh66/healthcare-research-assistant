from unittest.mock import MagicMock

from app.agents.nodes import claim_verifier
from app.agents.nodes.claim_verifier import CritiqueResult


def _state(answer="Widget therapy reduces symptoms [PMID 111].", verify_attempts=0):
    return {
        "final_answer": answer,
        "evidence": [{"pmid": "111", "claim": "c", "evidence": "e", "study_type": "RCT", "confidence": "high"}],
        "verify_attempts": verify_attempts,
        "plan": ["Start"],
    }


def _critique(supported, claims=None):
    return CritiqueResult(all_claims_supported=supported, unsupported_claims=claims or [])


def test_all_supported_appends_answer_and_ends(monkeypatch):
    monkeypatch.setattr(
        claim_verifier, "structured_critique_llm",
        MagicMock(invoke=MagicMock(return_value=_critique(True))),
    )

    result = claim_verifier.claim_verifier_node(_state())

    assert result["needs_regeneration"] is False
    assert result["messages"] == [{"role": "assistant", "content": _state()["final_answer"]}]
    assert result["plan"][-1] == "Claim verification: all claims supported"


def test_unsupported_with_budget_left_requests_regeneration(monkeypatch):
    monkeypatch.setattr(
        claim_verifier, "structured_critique_llm",
        MagicMock(invoke=MagicMock(return_value=_critique(False, ["'cures everything' unsupported"]))),
    )

    result = claim_verifier.claim_verifier_node(_state(verify_attempts=0))

    assert result["needs_regeneration"] is True
    assert result["verify_attempts"] == 1
    assert result["unsupported_claims"] == ["'cures everything' unsupported"]
    assert "messages" not in result  # not appended yet — responder will regenerate


def test_unsupported_after_budget_spent_attaches_caveat_and_ends(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(
        claim_verifier, "structured_critique_llm",
        MagicMock(invoke=MagicMock(return_value=_critique(False, ["'cures everything' unsupported"]))),
    )

    result = claim_verifier.claim_verifier_node(_state(verify_attempts=settings.CLAIM_VERIFY_MAX_RETRIES))

    assert result["needs_regeneration"] is False
    assert "### ⚠️ Self-Check Note" in result["final_answer"]
    assert result["messages"][0]["content"] == result["final_answer"]
    assert any("caveat attached" in note for note in result["plan"])
