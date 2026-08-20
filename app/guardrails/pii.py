import re

# Lightweight, dependency-free PII detection — deliberately regex-based rather
# than an NLP/NER model (e.g. Presidio+spaCy), which needs a large model loaded
# into memory for every request and can OOM on resource-constrained machines.
_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}

PII_REFUSAL_MESSAGE = (
    "I can't process that message because it appears to contain personal information "
    "(like an email, phone number, SSN, or card number). Please rephrase your question "
    "without including personal details."
)


def contains_pii(message: str) -> bool:
    return any(pattern.search(message) for pattern in _PII_PATTERNS.values())
