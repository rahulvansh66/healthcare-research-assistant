from typing import List
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 600
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text using LangChain's RecursiveCharacterTextSplitter, which tries
    a prioritized list of separators (paragraphs, lines, sentences, words)
    and falls back progressively so it never collapses into one giant chunk.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        splitter = (
            _splitter
            if chunk_size == CHUNK_SIZE and chunk_overlap == CHUNK_OVERLAP
            else RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )

        chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]
        logfire.info(f"✅ Generated {len(chunks)} chunks")
        return chunks
