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


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[List[dict], operator.add]
    current_query: str
    search_params: Optional[PubMedSearchParams]
    pubmed_search_requested: bool  # UI "PubMed Search" toggle for this turn
    fresh_search_requested: bool  # planner-detected dissatisfaction ("retry"/"search more")
    retrieval_source: str  # "live_pubmed" | "session_cache" | "none"
    documents: List[PubMedDocument]
    evidence: List[Evidence]
    citations: List[Citation]
    _background_store_payload: Optional[List[PubMedDocument]]  # top-N live results awaiting background storage
    plan: List[str]
    status: str
    final_answer: str
