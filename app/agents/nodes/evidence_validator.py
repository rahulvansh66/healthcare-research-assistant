from typing import Literal

import logfire
from pydantic import BaseModel

from app.agents.prompts.evidence import build_validation_prompt
from app.agents.state import AgentState
from app.config import settings
from app.gateway import get_langchain_llm

llm = get_langchain_llm(feature="evidence_validator")


class Coverage(BaseModel):
    intervention: bool
    population: bool
    outcome: bool


class EvidenceValidation(BaseModel):
    sufficient: bool
    reason: Literal["sufficient", "query_failure", "population_mismatch", "corpus_failure"]
    coverage: Coverage
    notes: str


structured_llm = llm.with_structured_output(EvidenceValidation)


def evidence_validator_node(state: AgentState) -> dict:
    """
    Claim-level sufficiency check. Reads the extracted evidence + the user's
    question and emits a typed verdict (``evidence_verdict``). route_validator in
    the graph turns that verdict into a routing decision — this node never
    decides what to do next itself.

    A failed direct-PMID lookup (no documents, ``direct_pmid`` set) is passed
    straight through as ``corpus_failure`` so route_validator can send it to the
    responder rather than the web fallback.
    """
    evidence = state.get("evidence", [])
    user_message = state["messages"][-1]["content"] if state["messages"] else state["current_query"]
    top_score = state.get("rerank_top_score", 0.0)

    if top_score:
        rerank_hint = (
            f"top rerank score {top_score:.2f} "
            f"({'above' if top_score >= settings.RERANK_RELEVANCE_THRESHOLD else 'below'} "
            f"the {settings.RERANK_RELEVANCE_THRESHOLD} relevance bar)"
        )
    else:
        rerank_hint = "no live search ran this turn (session cache or direct PMID)"

    evidence_block = "\n\n".join(
        f"[PMID {e['pmid']}] ({e['study_type']}, confidence={e['confidence']})\n"
        f"Claim: {e['claim']}\nSupporting text: {e['evidence']}"
        for e in evidence
    )

    prompt = build_validation_prompt(user_message, evidence_block, rerank_hint)

    with logfire.span("🔎 Evidence Validation"):
        verdict = structured_llm.invoke(prompt)

    logfire.info(f"Evidence verdict: sufficient={verdict.sufficient} reason={verdict.reason}")

    verdict_dict = verdict.model_dump()
    cov = verdict_dict["coverage"]
    return {
        "evidence_verdict": verdict_dict,
        "status": f"Evidence {'sufficient' if verdict.sufficient else 'insufficient'} ({verdict.reason}).",
        "plan": state["plan"]
        + [
            "Tool: evidence_validator",
            f"Verdict: {'sufficient' if verdict.sufficient else verdict.reason}",
            f"Coverage: intervention={cov['intervention']}, population={cov['population']}, outcome={cov['outcome']}",
        ],
    }
