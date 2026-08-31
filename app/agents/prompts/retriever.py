def build_query_rewrite_prompt(
    original_question: str,
    failed_query: str,
    attempt: int,
    validator_feedback: str = "",
) -> str:
    feedback_block = (
        f'\n    EVIDENCE VALIDATOR FEEDBACK (why the last results fell short):\n    "{validator_feedback}"\n'
        if validator_feedback
        else ""
    )
    return f"""
    You are Medico's Retrieval Corrector. A PubMed search for the query below returned
    results that did not sufficiently answer the user's original question (retry attempt
    {attempt}).

    ORIGINAL USER QUESTION:
    "{original_question}"

    PREVIOUS PUBMED SEARCH QUERY (too narrow, too broad, or otherwise off-target):
    "{failed_query}"
    {feedback_block}
    Task:
    Rewrite the search query so it is more likely to surface relevant PubMed literature for
    the original question. Consider broadening or narrowing scope, using different but
    clinically equivalent terminology (e.g. drug class vs. specific drug, synonym conditions),
    or dropping overly specific qualifiers. Keep it a concise, PubMed-friendly plain-text
    query (no boolean operators — those are added downstream).
    """
