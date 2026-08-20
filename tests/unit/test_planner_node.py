from unittest.mock import MagicMock

from app.agents.nodes import planner
from app.agents.nodes.planner import PlannerDecision


def _state(user_message, history=None):
    messages = (history or []) + [{"role": "user", "content": user_message}]
    return {"messages": messages, "plan": ["Start"]}


def test_conversational_intent_skips_retrieval(monkeypatch):
    decision = PlannerDecision(intent="CONVERSATIONAL")
    monkeypatch.setattr(planner, "structured_llm", MagicMock(invoke=MagicMock(return_value=decision)))

    result = planner.planner_node(_state("what did you just tell me"))

    assert result["current_query"] == "CONVERSATIONAL"
    assert result["search_params"] is None
    assert result["fresh_search_requested"] is False
    assert "Retrieval: Skipped" in result["plan"]


def test_clinical_intent_extracts_search_params(monkeypatch):
    decision = PlannerDecision(
        intent="CLINICAL",
        search_query="widget therapy efficacy",
        date_from="2020/01/01",
        date_to=None,
        publication_type="Randomized Controlled Trial",
        fresh_search_requested=False,
    )
    monkeypatch.setattr(planner, "structured_llm", MagicMock(invoke=MagicMock(return_value=decision)))

    result = planner.planner_node(_state("does widget therapy work, recent RCTs only"))

    assert result["current_query"] == "widget therapy efficacy"
    assert result["search_params"] == {
        "date_from": "2020/01/01",
        "date_to": None,
        "publication_type": "Randomized Controlled Trial",
    }
    assert result["fresh_search_requested"] is False


def test_clinical_intent_detects_fresh_search_request(monkeypatch):
    decision = PlannerDecision(
        intent="CLINICAL",
        search_query="widget therapy",
        fresh_search_requested=True,
    )
    monkeypatch.setattr(planner, "structured_llm", MagicMock(invoke=MagicMock(return_value=decision)))

    result = planner.planner_node(_state("no, search again for something more recent"))

    assert result["fresh_search_requested"] is True


def test_detect_direct_pmid_from_url():
    message = "generate a summary for this https://pubmed.ncbi.nlm.nih.gov/39796530/ paper"
    assert planner._detect_direct_pmid(message) == "39796530"


def test_detect_direct_pmid_from_explicit_mention():
    assert planner._detect_direct_pmid("summarize PMID 39796530") == "39796530"
    assert planner._detect_direct_pmid("summarize PMID:39796530") == "39796530"
    assert planner._detect_direct_pmid("summarize pmid:39796530") == "39796530"


def test_detect_direct_pmid_ignores_bare_numeric_message():
    assert planner._detect_direct_pmid("39796530") is None


def test_detect_direct_pmid_returns_first_match_only():
    message = "compare PMID 111111 with PMID 222222"
    assert planner._detect_direct_pmid(message) == "111111"


def test_planner_node_direct_pmid_skips_llm_call(monkeypatch):
    monkeypatch.setattr(
        planner,
        "structured_llm",
        MagicMock(invoke=MagicMock(side_effect=AssertionError("LLM should not be called"))),
    )

    result = planner.planner_node(_state("generate summary for PMID 39796530"))

    assert result["direct_pmid"] == "39796530"
    assert result["current_query"] == "PMID:39796530"
    assert result["search_params"] is None


def test_planner_node_resets_direct_pmid_on_normal_turn(monkeypatch):
    decision = PlannerDecision(intent="CONVERSATIONAL")
    monkeypatch.setattr(planner, "structured_llm", MagicMock(invoke=MagicMock(return_value=decision)))

    result = planner.planner_node(_state("what did you just tell me"))

    assert result["direct_pmid"] is None
