def build_query_rewrite_prompt(original_question: str, failed_query: str, attempt: int) -> str:
    return f"""
    You are Medico's Retrieval Corrector. A PubMed search for the query below returned
    results that were not relevant enough to the user's original question (retry attempt
    {attempt}).

    ORIGINAL USER QUESTION:
    "{original_question}"

    PREVIOUS PUBMED SEARCH QUERY (too narrow, too broad, or otherwise off-target):
    "{failed_query}"

    Task:
    Rewrite the search query so it is more likely to surface relevant PubMed literature for
    the original question. Consider broadening or narrowing scope, using different but
    clinically equivalent terminology (e.g. drug class vs. specific drug, synonym conditions),
    or dropping overly specific qualifiers. Keep it a concise, PubMed-friendly plain-text
    query (no boolean operators — those are added downstream).
    """
