from unittest.mock import MagicMock

from app.guardrails import rails


def test_guard_returns_false_when_rails_not_initialised(monkeypatch):
    monkeypatch.setattr(rails, "_rails", None)

    fired, response = rails.guard("hello")

    assert fired is False
    assert response is None


def test_guard_passes_clean_message(monkeypatch):
    mock_rails = MagicMock()
    mock_rails.generate.return_value = {"role": "assistant", "content": "Sure, here is clinical info."}
    monkeypatch.setattr(rails, "_rails", mock_rails)

    fired, response = rails.guard("what are the symptoms of diabetes")

    assert fired is False
    assert response is None


def test_guard_fires_on_off_topic_refusal(monkeypatch):
    mock_rails = MagicMock()
    mock_rails.generate.return_value = {
        "role": "assistant",
        "content": "I'm Medico, a healthcare research assistant focused on medical and clinical topics. "
                   "I can't help with that — but ask me anything about health, medicine, or clinical guidelines!",
    }
    monkeypatch.setattr(rails, "_rails", mock_rails)

    fired, response = rails.guard("tell me a joke")

    assert fired is True
    assert "can't help with that" in response


def test_guard_handles_non_dict_result(monkeypatch):
    mock_rails = MagicMock()
    mock_rails.generate.return_value = "Hello! I'm Medico, your healthcare research assistant. What can I help you with today?"
    monkeypatch.setattr(rails, "_rails", mock_rails)

    fired, response = rails.guard("hi")

    assert fired is True
