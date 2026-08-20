from typing import List, Literal

import logfire
from pydantic import BaseModel

from app.agents.prompts.evidence import build_evidence_prompt
from app.agents.state import AgentState
from app.config import settings
from app.gateway import get_langchain_llm

llm = get_langchain_llm(feature="evidence_agent")


class EvidenceRecord(BaseModel):
    pmid: str
    claim: str
    evidence: str
    study_type: str
    confidence: Literal["high", "moderate", "low"]


class EvidenceExtraction(BaseModel):
    records: List[EvidenceRecord]
    insufficient_evidence: bool


structured_llm = llm.with_structured_output(EvidenceExtraction)


def evidence_agent_node(state: AgentState) -> dict:
    """
    Turns retrieved PubMed abstracts into structured, PMID-grounded Evidence
    records. Never lets the LLM invent a citation: any record whose PMID isn't
    among the retrieved documents is dropped before it reaches state.
    """
    documents = state["documents"]

    if not documents:
        return {
            "evidence": [],
            "status": "No literature retrieved — insufficient evidence.",
            "plan": state["plan"] + ["Tool: evidence_agent", "Evidence: Insufficient (no documents)"],
        }

    user_message = state["messages"][-1]["content"] if state["messages"] else state["current_query"]

    # Cap how many documents' full text get inlined, regardless of how many
    # total documents are in play (a session-cache hit can carry more
    # documents than a live search does) — keeps the prompt/JSON-mode
    # generation bounded the same way on both retrieval paths.
    full_text_budget = settings.FULLTEXT_TOP_N

    def _doc_block(d: dict) -> str:
        nonlocal full_text_budget
        block = (
            f"PMID: {d['pmid']}\nTITLE: {d['title']}\nSTUDY TYPES: {', '.join(d['pub_types']) or 'Unknown'}\n"
            f"ABSTRACT: {d['abstract']}"
        )
        if d.get("full_text_markdown") and full_text_budget > 0:
            block += f"\nFULL TEXT: {d['full_text_markdown'][:settings.FULLTEXT_EVIDENCE_CHAR_LIMIT]}"
            full_text_budget -= 1
        return block

    docs_block = "\n\n".join(_doc_block(d) for d in documents)

    prompt = build_evidence_prompt(user_message, docs_block)

    with logfire.span("🧪 Evidence Extraction"):
        result = structured_llm.invoke(prompt)

    valid_pmids = {d["pmid"] for d in documents}
    records = [r for r in result.records if r.pmid in valid_pmids]
    dropped = len(result.records) - len(records)
    if dropped:
        logfire.warning(f"Dropped {dropped} evidence record(s) citing a PMID outside the retrieved set.")

    if result.insufficient_evidence or not records:
        return {
            "evidence": [],
            "status": "Retrieved literature does not sufficiently answer the question.",
            "plan": state["plan"] + ["Tool: evidence_agent", "Evidence: Insufficient"],
        }

    return {
        "evidence": [r.model_dump() for r in records],
        "status": "Evidence extracted.",
        "plan": state["plan"] + ["Tool: evidence_agent", f"Evidence Records: {len(records)}"],
    }
