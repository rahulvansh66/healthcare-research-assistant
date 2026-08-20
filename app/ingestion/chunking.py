from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import settings

_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
    strip_headers=False,
)


def chunk_markdown(markdown: str) -> list[dict]:
    """
    Header-aware split (keeps title/metadata/abstract and each full-text
    section together), then bounded by size. Each chunk carries its header
    path so similarity search retains section context even after the
    character splitter breaks a header section into multiple pieces.
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = []
    for header_doc in _HEADER_SPLITTER.split_text(markdown):
        header_path = " > ".join(
            v for v in (header_doc.metadata.get("Header 1"), header_doc.metadata.get("Header 2")) if v
        )
        for piece in char_splitter.split_text(header_doc.page_content):
            piece = piece.strip()
            if piece:
                chunks.append({"text": piece, "header_path": header_path})
    return chunks
