from app.services.retrieval.markdown_builder import build_article_markdown, build_full_text_markdown


def test_build_full_text_markdown_renders_sections_in_order():
    sections = [
        {"section": "INTRO", "text": "Intro text."},
        {"section": "METHODS", "text": "Methods text."},
    ]

    result = build_full_text_markdown(sections)

    assert result == "## Intro\nIntro text.\n\n## Methods\nMethods text."


def test_build_full_text_markdown_empty_list():
    assert build_full_text_markdown([]) == ""


def test_build_article_markdown_full_document(make_pubmed_document):
    doc = make_pubmed_document(
        title="A Study",
        journal="Journal X",
        year="2022",
        authors=["Smith J", "Doe A"],
        pub_types=["Randomized Controlled Trial"],
        abstract="This is the abstract.",
        full_text_markdown="## Results\nSome results.",
    )

    markdown = build_article_markdown(doc)

    assert markdown.startswith("# A Study")
    assert "**Journal:** Journal X" in markdown
    assert "**Year:** 2022" in markdown
    assert "**Authors:** Smith J, Doe A" in markdown
    assert "**Publication Types:** Randomized Controlled Trial" in markdown
    assert "### Abstract\nThis is the abstract." in markdown
    assert "## Results\nSome results." in markdown


def test_build_article_markdown_omits_missing_fields(make_pubmed_document):
    doc = make_pubmed_document(
        title="",
        journal="",
        year=None,
        authors=[],
        pub_types=[],
        abstract="",
        full_text_markdown=None,
    )

    markdown = build_article_markdown(doc)

    assert markdown == ""


def test_build_article_markdown_abstract_only(make_pubmed_document):
    doc = make_pubmed_document(
        title="Title Only",
        journal="",
        year=None,
        authors=[],
        pub_types=[],
        abstract="Just an abstract.",
        full_text_markdown=None,
    )

    markdown = build_article_markdown(doc)

    assert markdown == "# Title Only\n\n### Abstract\nJust an abstract."
