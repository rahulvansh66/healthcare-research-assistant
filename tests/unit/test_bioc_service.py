from unittest.mock import MagicMock

from app.services.retrieval import bioc_service


def _mock_response(json_data):
    resp = MagicMock()
    resp.json.return_value = json_data
    return resp


def test_returns_none_when_get_fails(monkeypatch):
    monkeypatch.setattr(bioc_service, "_get", lambda url, params: None)

    assert bioc_service.fetch_full_text_sections("12345") is None


def test_returns_none_on_malformed_json(monkeypatch):
    resp = MagicMock()
    resp.json.side_effect = ValueError("bad json")
    monkeypatch.setattr(bioc_service, "_get", lambda url, params: resp)

    assert bioc_service.fetch_full_text_sections("12345") is None


def test_returns_none_when_no_usable_sections(monkeypatch):
    data = [{"documents": [{"passages": [{"text": "", "infons": {"section_type": "INTRO"}}]}]}]
    monkeypatch.setattr(bioc_service, "_get", lambda url, params: _mock_response(data))

    assert bioc_service.fetch_full_text_sections("12345") is None


def test_skips_uninteresting_sections_and_merges_consecutive_same_section(monkeypatch):
    data = [{
        "documents": [{
            "passages": [
                {"text": "Title text", "infons": {"section_type": "TITLE"}},
                {"text": "Abstract text", "infons": {"section_type": "ABSTRACT"}},
                {"text": "Intro part 1.", "infons": {"section_type": "INTRO"}},
                {"text": "Intro part 2.", "infons": {"section_type": "INTRO"}},
                {"text": "Methods text.", "infons": {"section_type": "METHODS"}},
                {"text": "References...", "infons": {"section_type": "REF"}},
            ]
        }]
    }]
    monkeypatch.setattr(bioc_service, "_get", lambda url, params: _mock_response(data))

    sections = bioc_service.fetch_full_text_sections("12345")

    assert sections == [
        {"section": "INTRO", "text": "Intro part 1.\nIntro part 2."},
        {"section": "METHODS", "text": "Methods text."},
    ]


def test_single_collection_dict_is_treated_as_one_item_list(monkeypatch):
    data = {"documents": [{"passages": [{"text": "Body content", "infons": {"type": "Body"}}]}]}
    monkeypatch.setattr(bioc_service, "_get", lambda url, params: _mock_response(data))

    sections = bioc_service.fetch_full_text_sections("12345")

    assert sections == [{"section": "BODY", "text": "Body content"}]


def test_falls_back_to_body_when_no_section_type_infon(monkeypatch):
    data = [{"documents": [{"passages": [{"text": "Untyped content", "infons": {}}]}]}]
    monkeypatch.setattr(bioc_service, "_get", lambda url, params: _mock_response(data))

    sections = bioc_service.fetch_full_text_sections("12345")

    assert sections == [{"section": "BODY", "text": "Untyped content"}]
