from app.agents.nodes import web_search
from app.agents.state import WebResult


def _state(query="opening hours of the NIH library"):
    return {"current_query": query, "plan": ["Start"]}


def test_populates_web_results_and_sets_fallback_flag(monkeypatch):
    hits = [
        WebResult(title="NIH Library", url="https://nih.example/library", snippet="Open 9-5."),
        WebResult(title="Hours", url="https://nih.example/hours", snippet="Mon-Fri."),
    ]
    monkeypatch.setattr(web_search, "tavily_search", lambda q: hits)

    result = web_search.web_search_node(_state())

    assert result["web_results"] == hits
    assert result["used_web_fallback"] is True
    assert any("Web Results: 2" in note for note in result["plan"])


def test_empty_results_still_flags_fallback(monkeypatch):
    monkeypatch.setattr(web_search, "tavily_search", lambda q: [])

    result = web_search.web_search_node(_state())

    assert result["web_results"] == []
    assert result["used_web_fallback"] is True
    assert "found nothing" in result["status"]
