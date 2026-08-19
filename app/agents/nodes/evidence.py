from typing import List, Literal

import logfire
from pydantic import BaseModel

from app.agents.state import AgentState
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

    docs_block = "\n\n".join(
        f"PMID: {d['pmid']}\nTITLE: {d['title']}\nSTUDY TYPES: {', '.join(d['pub_types']) or 'Unknown'}\n"
        f"ABSTRACT: {d['abstract']}"
        for d in documents
    )

    prompt = f"""
    You are Medico's Evidence Agent. Extract structured evidence records STRICTLY from
    the PubMed abstracts below to help answer the user's question. Rules:
    - Only use the PMIDs and text given below — never invent a PMID or a claim not
      present in the supplied abstracts.
    - Distinguish study type where possible (e.g. Randomized Controlled Trial,
      Meta-Analysis, Observational, Systematic Review) using STUDY TYPES and the
      abstract's own description of its methodology.
    - Set confidence based on study type and how directly the abstract supports the claim
      (RCT/meta-analysis + direct match = high; observational or partial match = moderate;
      weak/indirect support = low).
    - One or more evidence records per document as warranted by distinct claims.
    - Set insufficient_evidence=true (and return an empty records list) if none of the
      abstracts actually answer the user's question.

    USER QUESTION:
    "{user_message}"

    RETRIEVED ABSTRACTS:
    {docs_block}
    """

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
