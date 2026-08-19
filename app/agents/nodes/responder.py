import logfire
from app.agents.state import AgentState
from app.config import settings
from app.gateway import portkey_client, extract_cache_status


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

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    evidence = state.get("evidence", [])

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are Medico, a friendly and knowledgeable healthcare research assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    elif not evidence:
        logfire.info("Generating insufficient-evidence response.")
        prompt = f"""
        You are Medico, a healthcare research assistant. A PubMed search was performed
        for the user's question, but the retrieved literature did not sufficiently
        answer it. Explain plainly that you could not find sufficient evidence for this
        specific question, invite the user to reformulate or narrow the question, and
        do not speculate or answer from general knowledge.

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """
    else:
        logfire.info("Generating evidence-grounded response.")
        evidence_block = "\n\n".join(
            f"[PMID {e['pmid']}] ({e['study_type']}, confidence={e['confidence']})\n"
            f"Claim: {e['claim']}\nSupporting text: {e['evidence']}"
            for e in evidence
        )

        prompt = f"""
        You are Medico, a healthcare research assistant. Answer ONLY using the EVIDENCE
        below. For every medical claim: cite the PMID, distinguish RCT vs
        observational/meta-analysis evidence, note important limitations, and do not
        invent evidence beyond what is given. If the evidence only partially covers the
        question, say so plainly rather than filling gaps with general knowledge.

        EVIDENCE:
        {evidence_block}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"

        End your answer with an "### Evidence" section listing each cited PMID and its
        study type.
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                model=f"@{settings.GROQ_SLUG}/openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

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
