from typing import List

import logfire
from langgraph.config import get_stream_writer
from pydantic import BaseModel

from app.agents.prompts.responder import build_critique_prompt
from app.agents.state import AgentState
from app.config import settings
from app.gateway import get_langchain_llm

critique_llm = get_langchain_llm(feature="claim_verifier")


class CritiqueResult(BaseModel):
    all_claims_supported: bool
    unsupported_claims: List[str]


structured_critique_llm = critique_llm.with_structured_output(CritiqueResult)


def _evidence_block(evidence: list[dict]) -> str:
    return "\n\n".join(
        f"[PMID {e['pmid']}] ({e['study_type']}, confidence={e['confidence']})\n"
        f"Claim: {e['claim']}\nSupporting text: {e['evidence']}"
        for e in evidence
    )


def claim_verifier_node(state: AgentState) -> dict:
    """
    Fact-checks the grounded draft answer against the evidence it was supposed to
    be built from. Only reached for evidence-grounded answers (see
    route_responder).

    - all claims supported  -> append the answer to the transcript, END.
    - unsupported claims, retry budget left -> flag them, bump verify_attempts,
      loop back to responder (which regenerates with a directive to drop them).
    - unsupported claims, budget spent -> append the '### ⚠️ Self-Check Note'
      caveat to the answer, append to the transcript, END.

    The assistant message is appended HERE (not in the responder) so the
    regenerate cycle doesn't leave duplicate turns in the checkpointed history.
    """
    content = state["final_answer"]
    evidence = state.get("evidence", [])
    verify_attempts = state.get("verify_attempts", 0)

    with logfire.span("🧐 Claim Verification"):
        critique = structured_critique_llm.invoke(
            build_critique_prompt(_evidence_block(evidence), content)
        )

    supported = critique.all_claims_supported or not critique.unsupported_claims

    if supported:
        logfire.info("✅ Claim verification: all claims supported.")
        return {
            "needs_regeneration": False,
            "unsupported_claims": [],
            "messages": [{"role": "assistant", "content": content}],
            "status": "Response generated.",
            "plan": state["plan"] + ["Claim verification: all claims supported"],
        }

    if verify_attempts < settings.CLAIM_VERIFY_MAX_RETRIES:
        logfire.warning(
            f"Claim verification flagged {len(critique.unsupported_claims)} unsupported "
            f"claim(s); regenerating (attempt {verify_attempts + 1})."
        )
        return {
            "needs_regeneration": True,
            "unsupported_claims": critique.unsupported_claims,
            "verify_attempts": verify_attempts + 1,
            "status": f"Regenerating answer — {len(critique.unsupported_claims)} unsupported claim(s).",
            "plan": state["plan"]
            + [f"Claim verification: flagged {len(critique.unsupported_claims)} claim(s) — regenerating"],
        }

    # Retry budget spent: keep the answer but attach the caveat.
    caveat_lines = "\n".join(f"- {c}" for c in critique.unsupported_claims)
    caveat = (
        "\n\n### ⚠️ Self-Check Note\nThe following statement(s) may not be fully supported "
        f"by the retrieved evidence:\n{caveat_lines}"
    )
    content_with_caveat = content + caveat
    try:
        get_stream_writer()(caveat)
    except Exception:
        pass  # no active custom stream (e.g. /query via .invoke())

    logfire.warning(
        f"Claim verification: {len(critique.unsupported_claims)} unsupported claim(s) "
        f"remain after {verify_attempts} regeneration attempt(s) — caveat attached."
    )
    return {
        "needs_regeneration": False,
        "unsupported_claims": critique.unsupported_claims,
        "final_answer": content_with_caveat,
        "messages": [{"role": "assistant", "content": content_with_caveat}],
        "status": "Response generated with self-check caveat.",
        "plan": state["plan"]
        + [f"Claim verification: {len(critique.unsupported_claims)} unsupported claim(s) — caveat attached"],
    }
