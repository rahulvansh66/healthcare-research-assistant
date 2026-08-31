from unittest.mock import MagicMock

from app.agents.nodes import query_rewriter
from app.agents.nodes.query_rewriter import QueryRewrite


def _state(**overrides):
    base = {
        "messages": [{"role": "user", "content": "does widget therapy work in children"}],
        "current_query": "widget therapy",
        "retrieval_attempts": 0,
        "evidence_verdict": {"reason": "query_failure", "notes": "too broad — narrow to pediatric cohorts"},
        "plan": ["Start"],
    }
    base.update(overrides)
    return base


def test_rewrites_query_and_bumps_attempts(monkeypatch):
    monkeypatch.setattr(
        query_rewriter, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=QueryRewrite(rewritten_query="widget therapy pediatric outcomes"))),
    )

    result = query_rewriter.query_rewriter_node(_state())

    assert result["current_query"] == "widget therapy pediatric outcomes"
    assert result["retrieval_attempts"] == 1
    assert any("query_rewriter" in note for note in result["plan"])


def test_second_cycle_increments_from_prior_attempts(monkeypatch):
    monkeypatch.setattr(
        query_rewriter, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=QueryRewrite(rewritten_query="v3"))),
    )

    result = query_rewriter.query_rewriter_node(_state(retrieval_attempts=1))

    assert result["retrieval_attempts"] == 2


def test_blank_rewrite_falls_back_to_previous_query(monkeypatch):
    monkeypatch.setattr(
        query_rewriter, "structured_llm",
        MagicMock(invoke=MagicMock(return_value=QueryRewrite(rewritten_query="   "))),
    )

    result = query_rewriter.query_rewriter_node(_state(current_query="widget therapy"))

    assert result["current_query"] == "widget therapy"


def test_validator_notes_are_passed_to_the_prompt(monkeypatch):
    captured = {}

    def _capture(prompt):
        captured["prompt"] = prompt
        return QueryRewrite(rewritten_query="x")

    monkeypatch.setattr(query_rewriter, "structured_llm", MagicMock(invoke=_capture))

    query_rewriter.query_rewriter_node(_state())

    assert "narrow to pediatric cohorts" in captured["prompt"]
