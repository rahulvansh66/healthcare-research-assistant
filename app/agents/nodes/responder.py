from typing import List

import logfire
from langgraph.config import get_stream_writer
from pydantic import BaseModel

from app.agents.history import format_history
from app.agents.prompts.responder import (
    build_conversational_prompt,
    build_critique_prompt,
    build_grounded_answer_prompt,
    build_insufficient_evidence_prompt,
)
from app.agents.state import AgentState
from app.config import settings
from app.gateway import get_langchain_llm, portkey_client, extract_cache_status

critique_llm = get_langchain_llm(feature="responder_critic")


class CritiqueResult(BaseModel):
    all_claims_supported: bool
    unsupported_claims: List[str]


structured_critique_llm = critique_llm.with_structured_output(CritiqueResult)


def _build_citations(evidence: list[dict], documents: list[dict]) -> list[dict]:
    """Derive citations deterministically from evidence + documents — never
    trust the LLM to reproduce them correctly in prose."""
    docs_by_pmid = {d["pmid"]: d for d in documents}
    seen = set()
    citations = []
    for e in evidence:
        pmid = e["pmid"]
        if pmid in seen:
            continue
        seen.add(pmid)
        d = docs_by_pmid.get(pmid, {})
        citations.append({
            "pmid": pmid,
            "title": d.get("title", ""),
            "journal": d.get("journal", ""),
            "year": d.get("year"),
            "study_type": e["study_type"],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return citations


def generate_node(state: AgentState):
    """
    Synthesizes a response from conversation history and, for clinical queries,
    strictly from the structured Evidence extracted upstream. Uses the native
    Portkey client (not LangChain) so we can read the x-portkey-cache-status
    response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = format_history(state["messages"][:-1], settings.MAX_HISTORY_CHARS)

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    evidence = state.get("evidence", [])

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = build_conversational_prompt(history_str, user_msg)
    elif not evidence:
        logfire.info("Generating insufficient-evidence response.")
        prompt = build_insufficient_evidence_prompt(history_str, user_msg)
    else:
        logfire.info("Generating evidence-grounded response.")
        evidence_block = "\n\n".join(
            f"[PMID {e['pmid']}] ({e['study_type']}, confidence={e['confidence']})\n"
            f"Claim: {e['claim']}\nSupporting text: {e['evidence']}"
            for e in evidence
        )

        prompt = build_grounded_answer_prompt(evidence_block, history_str, user_msg)

    with logfire.span("✍️ LLM Synthesis"):
        try:
            # stream=True unconditionally: get_stream_writer() no-ops when the graph
            # runs via .invoke() (e.g. /query), so this is safe for both callers —
            # only .stream(..., stream_mode="custom") callers (/query/stream) see tokens live.
            stream = portkey_client.chat.completions.create(
                model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.RESPONDER_TEMPERATURE,
                stream=True,
            )
            cache_status = extract_cache_status(stream)
            is_cache_hit = cache_status == "HIT"

            writer = get_stream_writer()
            content_parts = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    content_parts.append(delta)
                    writer(delta)
            content = "".join(content_parts)

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            if evidence:
                with logfire.span("🧐 Self-Critique"):
                    critique = structured_critique_llm.invoke(build_critique_prompt(evidence_block, content))

                if not critique.all_claims_supported and critique.unsupported_claims:
                    caveat_lines = "\n".join(f"- {c}" for c in critique.unsupported_claims)
                    caveat = f"\n\n### ⚠️ Self-Check Note\nThe following statement(s) may not be fully supported by the retrieved evidence:\n{caveat_lines}"
                    content += caveat
                    writer(caveat)
                    plan_update = plan_update + [f"Self-critique: flagged {len(critique.unsupported_claims)} unsupported claim(s)"]
                    logfire.warning(f"Self-critique flagged {len(critique.unsupported_claims)} unsupported claim(s).")
                else:
                    plan_update = plan_update + ["Self-critique: all claims supported"]

            citations = _build_citations(evidence, state.get("documents", [])) if evidence else []

            return {
                "final_answer": content,
                "citations": citations,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
