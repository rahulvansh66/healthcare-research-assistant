from app.config import settings
from app.ingestion.chunking import chunk_markdown


def test_splits_on_headers_and_keeps_header_path():
    markdown = "# Title\nIntro line.\n\n## Abstract\nSome abstract text."
    chunks = chunk_markdown(markdown)

    assert [c["header_path"] for c in chunks] == ["Title", "Title > Abstract"]
    assert chunks[0]["text"] == "# Title\nIntro line."
    assert chunks[1]["text"] == "## Abstract\nSome abstract text."


def test_no_headers_yields_single_header_path_empty():
    chunks = chunk_markdown("Just plain text with no headers at all.")

    assert len(chunks) == 1
    assert chunks[0]["header_path"] == ""
    assert chunks[0]["text"] == "Just plain text with no headers at all."


def test_long_section_is_split_by_character_splitter_and_keeps_header_path():
    long_body = "sentence. " * (settings.CHUNK_SIZE // len("sentence. ") + 20)
    markdown = f"# Title\n{long_body}"

    chunks = chunk_markdown(markdown)

    assert len(chunks) > 1
    assert all(c["header_path"] == "Title" for c in chunks)
    assert all(len(c["text"]) <= settings.CHUNK_SIZE for c in chunks)


def test_blank_markdown_yields_no_chunks():
    assert chunk_markdown("") == []


def test_whitespace_only_pieces_are_dropped():
    markdown = "# Title\n   \n\n## Section\nreal content"
    chunks = chunk_markdown(markdown)

    assert all(c["text"].strip() != "" for c in chunks)
