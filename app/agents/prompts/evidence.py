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
