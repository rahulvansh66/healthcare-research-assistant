from typing import TypedDict, List, Optional, Annotated
import operator


class PubMedSearchParams(TypedDict, total=False):
    date_from: Optional[str]
    date_to: Optional[str]
    publication_type: Optional[str]


class PubMedDocument(TypedDict):
    pmid: str
    title: str
    abstract: str
    journal: str
    year: Optional[str]
    pub_types: List[str]
    authors: List[str]
    has_full_text: bool
    full_text_markdown: Optional[str]


class Evidence(TypedDict):
    pmid: str
    claim: str
    evidence: str
    study_type: str
    confidence: str  # "high" | "moderate" | "low"


class Citation(TypedDict):
    pmid: str
    title: str
    journal: str
    year: Optional[str]
    study_type: str
    url: str


class WebResult(TypedDict):
    title: str
    url: str
    snippet: str


class EvidenceVerdict(TypedDict, total=False):
    sufficient: bool
    reason: str  # "sufficient" | "query_failure" | "population_mismatch" | "corpus_failure"
    coverage: dict  # {"intervention": bool, "population": bool, "outcome": bool}
    notes: str


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[List[dict], operator.add]
    current_query: str
    search_params: Optional[PubMedSearchParams]
    pubmed_search_requested: bool  # UI "PubMed Search" toggle for this turn
    fresh_search_requested: bool  # planner-detected dissatisfaction ("retry"/"search more")
    direct_pmid: Optional[str]  # set by planner when a PMID/URL was explicitly detected; None otherwise
    retrieval_source: str  # "live_pubmed" | "session_cache" | "none"
    retrieval_attempts: int  # retriever -> query_rewriter -> retriever cycles used this turn (bounded by CRAG_MAX_RETRIES)
    rerank_top_score: float  # top Jina rerank score from the most recent live search (0.0 if none); a hint for the validator
    documents: List[PubMedDocument]
    evidence: List[Evidence]
    evidence_verdict: Optional[EvidenceVerdict]  # evidence_validator's typed sufficiency verdict; drives route_validator
    citations: List[Citation]
    web_results: Optional[List[WebResult]]  # Tavily results from the corpus_failure fallback
    used_web_fallback: bool  # set by web_search node; steers responder + surfaced by main.py as web_sources
    verify_attempts: int  # responder -> claim_verifier -> responder cycles used this turn (bounded by CLAIM_VERIFY_MAX_RETRIES)
    unsupported_claims: List[str]  # claim_verifier's flagged claims, fed into the regeneration directive
    needs_regeneration: bool  # claim_verifier -> route_claim_verifier signal: loop back to responder vs. END
    _background_store_payload: Optional[List[PubMedDocument]]  # top-N live results awaiting background storage
    plan: List[str]
    status: str
    final_answer: str
