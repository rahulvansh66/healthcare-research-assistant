def build_evidence_prompt(user_message: str, docs_block: str) -> str:
    return f"""
    You are Medico's Evidence Agent. Extract structured evidence records STRICTLY from
    the PubMed abstracts (and full text, when provided) below to help answer the user's
    question. Rules:
    - Only use the PMIDs and text given below — never invent a PMID or a claim not
      present in the supplied abstract or full text.
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

    RETRIEVED LITERATURE:
    {docs_block}
    """


def build_validation_prompt(user_message: str, evidence_block: str, rerank_hint: str) -> str:
    return f"""
    You are Medico's Evidence Validator. Decide whether the EXTRACTED EVIDENCE below is
    sufficient to answer the user's question — a control-flow decision, not answer writing.

    Judge coverage on three axes and report each as true/false:
    - intervention: does the evidence concern the drug / test / procedure the user asked about?
    - population: does it concern the SAME population (age group, comorbidity, disease vs.
      non-disease state, etc.) the user asked about?
    - outcome: does it report the outcome the user asked about (mortality, remission, adverse
      event, etc.)?

    Then set `sufficient` and pick exactly one `reason`:
    - "sufficient": all three axes are covered and the evidence directly addresses the question.
    - "query_failure": the evidence is broadly on-topic but misses the specific question; a
      better PubMed search query could plausibly surface what's needed.
    - "population_mismatch": the evidence answers a DIFFERENT population/intervention/outcome
      than asked (e.g. user asks about metformin in NON-diabetic patients but every paper is in
      type-2 diabetes). A rewritten query could still fix this.
    - "corpus_failure": the question is unlikely to be answerable from PubMed at all (no such
      literature exists, or it's a logistics / guidelines-access / non-research question).

    Put a one-line justification in `notes`.

    RERANKER SIGNAL: {rerank_hint}

    USER QUESTION:
    "{user_message}"

    EXTRACTED EVIDENCE:
    {evidence_block if evidence_block.strip() else "(no evidence records were extracted)"}
    """
