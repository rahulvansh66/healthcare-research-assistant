from app.agents.state import PubMedDocument


def build_full_text_markdown(sections: list[dict]) -> str:
    """Render BioC sections as '## Section' markdown blocks, in order."""
    return "\n\n".join(f"## {s['section'].title()}\n{s['text']}" for s in sections)


def build_article_markdown(doc: PubMedDocument) -> str:
    """
    Organize an article's metadata, abstract, and (if present) full text into
    markdown for embedding. Any field that's missing or empty is omitted
    rather than rendered as a blank line.
    """
    lines = [f"# {doc['title']}"] if doc.get("title") else []

    meta_lines = []
    if doc.get("journal"):
        meta_lines.append(f"**Journal:** {doc['journal']}")
    if doc.get("year"):
        meta_lines.append(f"**Year:** {doc['year']}")
    if doc.get("authors"):
        meta_lines.append(f"**Authors:** {', '.join(doc['authors'])}")
    if doc.get("pub_types"):
        meta_lines.append(f"**Publication Types:** {', '.join(doc['pub_types'])}")
    if meta_lines:
        lines.append("\n".join(meta_lines))

    if doc.get("abstract"):
        lines.append(f"### Abstract\n{doc['abstract']}")

    if doc.get("full_text_markdown"):
        lines.append(doc["full_text_markdown"])

    return "\n\n".join(lines)
