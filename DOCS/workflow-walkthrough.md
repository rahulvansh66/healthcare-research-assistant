# Workflow walkthrough — 3 queries, node by node

Three example queries traced through the whole LangGraph agent, showing what each node
**reads**, **does**, and **writes** to the shared `AgentState`.

- Example 1 — a plain clinical question, happy path (no loops), plus a conversational follow-up.
- Example 2 — a population-mismatch question that triggers the CRAG query-rewrite loop and then
  the web-search fallback.
- Example 3 — a "summarise this PMID" request that hits the direct-PMID path and one
  claim-verifier regeneration.

> These traces are **illustrative** — realistic shapes and values, not captured from a live run.
> PMIDs / abstract text / scores are invented. Config values are the real defaults from
> `config/dev.yaml` (`RERANK_RELEVANCE_THRESHOLD=0.4`, `CRAG_MAX_RETRIES=2`,
> `CLAIM_VERIFY_MAX_RETRIES=1`, `EVIDENCE_TOP_N=5`, `FULLTEXT_TOP_N=5`, `PUBMED_FETCH_LIMIT=20`,
> `PUBMED_STORE_TOP_N=10`, `SESSION_CACHE_RELEVANCE_THRESHOLD=0.35`).

## Node contracts (quick reference)

| Node | File | LLM / API call | Structured output |
|---|---|---|---|
| guardrails gate | `app/guardrails/rails.py` (+ `pii.py`) | regex PII + NeMo (`allam-2-7b`) | — (returns `(rail_fired, rail_response)`) |
| `planner` | `nodes/planner.py` | 1 LLM call (skipped if a PMID/URL is regex-detected) | `PlannerDecision{intent, search_query, date_from, date_to, publication_type, fresh_search_requested}` |
| `retriever` | `nodes/retriever.py` | NCBI ESearch+EFetch, Jina rerank, NCBI BioC full text | — |
| `evidence_extractor` | `nodes/evidence_extractor.py` | 1 LLM call | `EvidenceExtraction{records: [EvidenceRecord{pmid, claim, evidence, study_type, confidence}], insufficient_evidence}` |
| `evidence_validator` | `nodes/evidence_validator.py` | 1 LLM call | `EvidenceValidation{sufficient, reason, coverage{intervention, population, outcome}, notes}` |
| `query_rewriter` | `nodes/query_rewriter.py` | 1 LLM call | `QueryRewrite{rewritten_query}` |
| `web_search` | `nodes/web_search.py` | Tavily search | — (`WebResult{title, url, snippet}` list) |
| `responder` | `nodes/responder.py` | 1 streaming LLM call | plain text; `citations` derived in Python |
| `claim_verifier` | `nodes/claim_verifier.py` | 1 LLM call | `CritiqueResult{all_claims_supported, unsupported_claims}` |

`AgentState` keys touched below: `messages`, `current_query`, `search_params`, `direct_pmid`,
`retrieval_source`, `retrieval_attempts`, `rerank_top_score`, `documents`, `evidence`,
`evidence_verdict`, `citations`, `web_results`, `used_web_fallback`, `verify_attempts`,
`unsupported_claims`, `needs_regeneration`, `plan`, `status`, `final_answer`.

---

# Example 1 — plain clinical question (happy path)

**Request**

```json
POST /query
{ "q": "What are the first-line pharmacological treatments for type 2 diabetes?",
  "thread_id": "demo-1", "pubmed_search_requested": false }
```

**Initial state** (`_build_initial_state` in `app/main.py`)

```json
{
  "messages": [{"role": "user", "content": "What are the first-line pharmacological treatments for type 2 diabetes?"}],
  "current_query": "What are the first-line pharmacological treatments for type 2 diabetes?",
  "search_params": null, "direct_pmid": null,
  "retrieval_source": "none", "retrieval_attempts": 0, "rerank_top_score": 0.0,
  "documents": [], "evidence": [], "evidence_verdict": null, "citations": [],
  "web_results": null, "used_web_fallback": false,
  "verify_attempts": 0, "unsupported_claims": [], "needs_regeneration": false,
  "plan": ["Start"], "status": "Initializing Graph..."
}
```

### 1. Guardrails gate  (`main.py` → `guard(q)`, before the graph)

- **in:** raw `q`
- **does:** regex PII scan (email / phone / SSN / card) → clean; NeMo intent rails + `self check
  input` → on-topic clinical question, no jailbreak.
- **out:** `(rail_fired=False, rail_response=None)` → graph runs.

### 2. `planner`

- **reads:** `messages` (history = `[]`, latest = the question)
- **does:** no PMID/URL in the text → 1 structured LLM call.
  `PlannerDecision(intent="CLINICAL", search_query="first-line pharmacological treatment type 2 diabetes", date_from=null, date_to=null, publication_type=null, fresh_search_requested=false)`
- **writes:**

```json
{ "current_query": "first-line pharmacological treatment type 2 diabetes",
  "search_params": {"date_from": null, "date_to": null, "publication_type": null},
  "fresh_search_requested": false, "direct_pmid": null,
  "status": "Researching clinical evidence for: first-line pharmacological treatment type 2 diabetes",
  "plan": ["Start", "Intent: Clinical Research", "Search Term: first-line pharmacological treatment type 2 diabetes"] }
```

- **route:** `route_planner` → `current_query != "CONVERSATIONAL"` → **retriever**.

### 3. `retriever`

- **reads:** `current_query`, `search_params`, `direct_pmid=None`, `messages` (len 1 →
  `first_turn=True`), `retrieval_attempts=0`
- **does:** `first_turn` → live search (session cache skipped).
  1. `search_pubmed(...)` → 18 PMIDs.
  2. `get_pubmed_articles(...)` → 18 `PubMedDocument`s.
  3. Jina rerank on `"title\nabstract"` → top score **0.71** (≥ 0.4, fine).
  4. Full text (BioC-PMC) fetched for the top 5; 2 are open-access.
  5. Keep top `EVIDENCE_TOP_N=5` for this turn; top `PUBMED_STORE_TOP_N=10` staged for the
     background Qdrant write.
- **writes:**

```json
{ "documents": [
    {"pmid": "40010001", "title": "Metformin as first-line therapy in type 2 diabetes: a systematic review",
     "abstract": "Across 34 RCTs, metformin monotherapy lowered HbA1c by ~1.1% ...",
     "journal": "Diabetes Care", "year": "2023", "pub_types": ["Systematic Review", "Meta-Analysis"],
     "authors": ["Ng K", "Patel R"], "has_full_text": true, "full_text_markdown": "## Introduction\n..."},
    {"pmid": "40010002", "title": "ADA/EASD consensus: pharmacologic management of hyperglycemia",
     "abstract": "Guideline recommends metformin plus lifestyle as initial therapy; SGLT2i or GLP-1 RA ...",
     "journal": "Diabetologia", "year": "2022", "pub_types": ["Guideline"], "authors": ["Davies M"],
     "has_full_text": false, "full_text_markdown": null},
    "... 3 more ..."
  ],
  "retrieval_source": "live_pubmed", "rerank_top_score": 0.71,
  "_background_store_payload": ["... top 10 reranked docs ..."],
  "status": "Found supporting literature.",
  "plan": ["...", "Tool: search_pubmed", "PMIDs Found: 18 (attempt 1)",
           "Top Relevance: 0.71", "Reranked: top 5 of 10 stored", "Full Text: 2/5"] }
```

- **route:** static edge → **evidence_extractor**.

### 4. `evidence_extractor`

- **reads:** `documents`, latest user message
- **does:** builds one text block per doc (`PMID / TITLE / STUDY TYPES / ABSTRACT` + full text for
  ≤ 5), 1 structured LLM call → `EvidenceExtraction`. Every returned `pmid` is checked against the
  retrieved set (hallucination guard) — none dropped here.
- **writes:**

```json
{ "evidence": [
    {"pmid": "40010001", "claim": "Metformin monotherapy is effective first-line therapy, lowering HbA1c ~1.1%.",
     "evidence": "Across 34 RCTs, metformin monotherapy lowered HbA1c by ~1.1% vs placebo.",
     "study_type": "Meta-Analysis", "confidence": "high"},
    {"pmid": "40010002", "claim": "Guidelines recommend metformin + lifestyle as initial therapy.",
     "evidence": "Guideline recommends metformin plus lifestyle as initial therapy; SGLT2i or GLP-1 RA when ASCVD/CKD/HF present.",
     "study_type": "Guideline", "confidence": "high"},
    {"pmid": "40010005", "claim": "SGLT2 inhibitors are preferred first-line with established heart failure.",
     "evidence": "In patients with HFrEF, SGLT2i reduced hospitalization regardless of diabetes status.",
     "study_type": "Randomized Controlled Trial", "confidence": "moderate"}
  ],
  "status": "Evidence extracted.",
  "plan": ["...", "Tool: evidence_extractor", "Evidence Records: 3"] }
```

- **route:** static edge → **evidence_validator**.

### 5. `evidence_validator`

- **reads:** latest user message, `evidence`, `rerank_top_score=0.71`
- **does:** 1 structured LLM call with the evidence block + a rerank hint
  (`"top rerank score 0.71 (above the 0.4 relevance bar)"`).
- **writes:**

```json
{ "evidence_verdict": {
    "sufficient": true, "reason": "sufficient",
    "coverage": {"intervention": true, "population": true, "outcome": true},
    "notes": "Evidence directly covers first-line drug choice for T2DM in the general adult population." },
  "status": "Evidence sufficient (sufficient).",
  "plan": ["...", "Tool: evidence_validator", "Verdict: sufficient",
           "Coverage: intervention=True, population=True, outcome=True"] }
```

- **route:** `route_validator` → `verdict.sufficient` → **responder**.

### 6. `responder`

- **reads:** `current_query` (not `"CONVERSATIONAL"`), `messages` (history empty), `evidence`,
  `used_web_fallback=False`, `verify_attempts=0`
- **does:** grounded branch → `build_grounded_answer_prompt(evidence_block, history, user_msg)`
  (no regeneration directive). Streaming LLM call; `extract_cache_status` → `MISS`.
  `_build_citations(evidence, documents)` dedupes by PMID.
- **writes** (note: **no `messages`** on the grounded path — `claim_verifier` appends it):

```json
{ "final_answer": "First-line pharmacological therapy for type 2 diabetes is **metformin**, combined with lifestyle measures [PMID 40010002]. A meta-analysis of 34 RCTs found metformin monotherapy lowers HbA1c by roughly 1.1% [PMID 40010001]. In patients with established heart failure, chronic kidney disease, or atherosclerotic cardiovascular disease, an SGLT2 inhibitor or GLP-1 receptor agonist is preferred first-line or added early [PMID 40010002, PMID 40010005].\n\n### Evidence\n- PMID 40010001 — Meta-Analysis\n- PMID 40010002 — Guideline\n- PMID 40010005 — Randomized Controlled Trial",
  "citations": [
    {"pmid": "40010001", "title": "Metformin as first-line therapy ...", "journal": "Diabetes Care", "year": "2023", "study_type": "Meta-Analysis", "url": "https://pubmed.ncbi.nlm.nih.gov/40010001/"},
    {"pmid": "40010002", "title": "ADA/EASD consensus ...", "journal": "Diabetologia", "year": "2022", "study_type": "Guideline", "url": "https://pubmed.ncbi.nlm.nih.gov/40010002/"},
    {"pmid": "40010005", "title": "SGLT2 inhibition in heart failure ...", "journal": "N Engl J Med", "year": "2021", "study_type": "Randomized Controlled Trial", "url": "https://pubmed.ncbi.nlm.nih.gov/40010005/"}
  ],
  "status": "Response generated.",
  "plan": ["..."] }
```

- **route:** `route_responder` → `is_grounded_answer(state)` is `True` → **claim_verifier**.

### 7. `claim_verifier`

- **reads:** `final_answer`, `evidence`, `verify_attempts=0`
- **does:** 1 structured LLM call — `build_critique_prompt(evidence_block, final_answer)` →
  `CritiqueResult(all_claims_supported=true, unsupported_claims=[])`.
- **writes:**

```json
{ "needs_regeneration": false, "unsupported_claims": [],
  "messages": [{"role": "assistant", "content": "First-line pharmacological therapy ... Randomized Controlled Trial"}],
  "status": "Response generated.",
  "plan": ["...", "Claim verification: all claims supported"] }
```

- **route:** `route_claim_verifier` → not `needs_regeneration` → **END**.

### Final API response (`main.py`)

```json
{ "question": "What are the first-line pharmacological treatments for type 2 diabetes?",
  "answer": "First-line pharmacological therapy for type 2 diabetes is **metformin** ... Randomized Controlled Trial",
  "thought_process": ["Start", "Intent: Clinical Research", "Search Term: ...",
    "Tool: search_pubmed", "PMIDs Found: 18 (attempt 1)", "Top Relevance: 0.71",
    "Reranked: top 5 of 10 stored", "Full Text: 2/5",
    "Tool: evidence_extractor", "Evidence Records: 3",
    "Tool: evidence_validator", "Verdict: sufficient", "Coverage: intervention=True, population=True, outcome=True",
    "Claim verification: all claims supported"],
  "status": "Response generated.",
  "sources": ["... the 5 PubMedDocuments ..."],
  "citations": ["... 3 Citation objects ..."],
  "web_sources": [] }
```

After the response is sent, `main.py` schedules the background job: embed the 10 staged docs and
upsert them into Qdrant filtered by `thread_id="demo-1"`.

### Follow-up turn (same thread) — conversational

**Request:** `{ "q": "put that in one sentence", "thread_id": "demo-1" }`

- **guardrails:** pass.
- **planner:** history now has the Q + grounded A. `PlannerDecision(intent="CONVERSATIONAL")` →
  writes `current_query="CONVERSATIONAL"`, `plan += ["Intent: Conversational/Memory", "Retrieval: Skipped"]`.
- **route_planner:** `"CONVERSATIONAL"` → **responder** (retriever / extractor / validator all skipped).
- **responder:** conversational branch → `build_conversational_prompt(history, "put that in one sentence")`.
  Not grounded → returns `final_answer` **and** `messages: [{"role":"assistant", ...}]`.
  `citations: []`.
- **route_responder:** `is_grounded_answer` is `False` (no evidence this turn) → **END**. No claim verifier.

```json
{ "answer": "Start type 2 diabetes treatment with metformin plus lifestyle change, adding an SGLT2 inhibitor or GLP-1 receptor agonist early when heart, kidney, or cardiovascular disease is present.",
  "thought_process": ["Start", "Intent: Conversational/Memory", "Retrieval: Skipped"],
  "citations": [], "sources": [], "web_sources": [] }
```

---

# Example 2 — population mismatch → CRAG rewrite → web fallback

**Request**

```json
POST /query
{ "q": "Does metformin reduce cardiovascular mortality in non-diabetic adults?",
  "thread_id": "demo-2", "pubmed_search_requested": false }
```

### 1. Guardrails — pass.

### 2. `planner`

`PlannerDecision(intent="CLINICAL", search_query="metformin cardiovascular mortality non-diabetic", fresh_search_requested=false)`

```json
{ "current_query": "metformin cardiovascular mortality non-diabetic",
  "search_params": {"date_from": null, "date_to": null, "publication_type": null},
  "plan": ["Start", "Intent: Clinical Research", "Search Term: metformin cardiovascular mortality non-diabetic"] }
```

→ **retriever**.

### 3. `retriever` — pass 1

- `first_turn=True` → live search. 15 PMIDs → rerank top score **0.63** (≥ 0.4, so the retriever
  is satisfied on relevance grounds). Top 5 kept.
- Every hit is about **type-2-diabetic** cohorts — relevant to "metformin + CV mortality" but not
  to the "non-diabetic" population.

```json
{ "documents": [
    {"pmid": "40020001", "title": "Metformin and cardiovascular outcomes in type 2 diabetes (UKPDS follow-up)",
     "abstract": "In overweight patients with type 2 diabetes, metformin reduced all-cause and CV mortality (HR 0.64) ...",
     "journal": "Lancet", "year": "2019", "pub_types": ["Randomized Controlled Trial"], "authors": ["Holman R"],
     "has_full_text": true, "full_text_markdown": "## Methods\n..."},
    {"pmid": "40020002", "title": "Metformin vs sulfonylurea and CV death in T2DM: cohort study",
     "abstract": "Among 92,000 adults with type 2 diabetes, metformin was associated with lower CV mortality ...",
     "journal": "Diabetologia", "year": "2020", "pub_types": ["Observational Study"], "authors": ["Roos S"],
     "has_full_text": false, "full_text_markdown": null},
    "... 3 more, all T2DM ..."
  ],
  "retrieval_source": "live_pubmed", "rerank_top_score": 0.63,
  "_background_store_payload": ["..."],
  "status": "Found supporting literature.",
  "plan": ["...", "Tool: search_pubmed", "PMIDs Found: 15 (attempt 1)", "Top Relevance: 0.63",
           "Reranked: top 5 of 10 stored", "Full Text: 3/5"] }
```

→ **evidence_extractor**.

### 4. `evidence_extractor` — pass 1

```json
{ "evidence": [
    {"pmid": "40020001", "claim": "Metformin lowered cardiovascular mortality (HR 0.64) in overweight type 2 diabetes.",
     "evidence": "In overweight patients with type 2 diabetes, metformin reduced all-cause and CV mortality (HR 0.64).",
     "study_type": "Randomized Controlled Trial", "confidence": "high"},
    {"pmid": "40020002", "claim": "Metformin use was associated with lower CV mortality vs sulfonylurea in type 2 diabetes.",
     "evidence": "Among 92,000 adults with type 2 diabetes, metformin was associated with lower CV mortality.",
     "study_type": "Observational Study", "confidence": "moderate"}
  ],
  "status": "Evidence extracted.",
  "plan": ["...", "Tool: evidence_extractor", "Evidence Records: 2"] }
```

→ **evidence_validator**.

### 5. `evidence_validator` — pass 1  →  `population_mismatch`

- **reads:** question ("non-diabetic adults"), `evidence` (all type-2 diabetes), `rerank_top_score=0.63`
- **does:** 1 LLM call. Intervention and outcome are covered, but population is **not** — every record is in a
  diabetic cohort.

```json
{ "evidence_verdict": {
    "sufficient": false, "reason": "population_mismatch",
    "coverage": {"intervention": true, "population": false, "outcome": true},
    "notes": "All evidence is in type-2-diabetic populations; the question asks specifically about non-diabetic adults." },
  "status": "Evidence insufficient (population_mismatch).",
  "plan": ["...", "Tool: evidence_validator", "Verdict: population_mismatch",
           "Coverage: intervention=True, population=False, outcome=True"] }
```

- **route:** `route_validator` → `reason ∈ {query_failure, population_mismatch}` **and**
  `retrieval_attempts(0) < CRAG_MAX_RETRIES(2)` → **query_rewriter**.

### 6. `query_rewriter`

- **reads:** original question (`messages[-1]`), failed query (`current_query`),
  `retrieval_attempts=0`, `evidence_verdict.notes`
- **does:** 1 structured LLM call — `QueryRewrite(rewritten_query="metformin cardiovascular mortality prediabetes OR non-diabetic OR insulin resistance randomized trial")`
- **writes:**

```json
{ "current_query": "metformin cardiovascular mortality prediabetes OR non-diabetic OR insulin resistance randomized trial",
  "retrieval_attempts": 1,
  "status": "Rewriting search query (attempt 1): metformin cardiovascular mortality prediabetes ...",
  "plan": ["...", "Tool: query_rewriter",
           "CRAG attempt 1: \"metformin cardiovascular mortality non-diabetic\" -> \"metformin cardiovascular mortality prediabetes OR non-diabetic OR insulin resistance randomized trial\""] }
```

- **route:** static edge → **retriever** (the cycle).

### 3b. `retriever` — pass 2

- `retrieval_attempts(1) > 0` → `on_rewrite_cycle=True` → forces a fresh live search with the new
  `current_query` (session cache is bypassed).
- The rewritten query surfaces a few prevention trials, but reranking is weak — top score **0.29**
  (< 0.4). Documents are still returned (the retriever no longer loops on low relevance; it just
  reports the score).

```json
{ "documents": [
    {"pmid": "40020050", "title": "Metformin for primary prevention of CVD in prediabetes: exploratory analysis",
     "abstract": "In the Diabetes Prevention Program, metformin did not significantly change cardiovascular event rates over 15 years ...",
     "journal": "Circulation", "year": "2017", "pub_types": ["Randomized Controlled Trial"], "authors": ["Goldberg R"],
     "has_full_text": false, "full_text_markdown": null}
  ],
  "retrieval_source": "live_pubmed", "rerank_top_score": 0.29,
  "plan": ["...", "Tool: search_pubmed", "PMIDs Found: 6 (attempt 2)", "Top Relevance: 0.29",
           "Reranked: top 1 of 1 stored", "Full Text: 0/1"] }
```

→ **evidence_extractor**.

### 4b. `evidence_extractor` — pass 2

The one weakly-related paper is about **prediabetes** and reports a **null** result; the model
returns `insufficient_evidence=true` (or a single low-confidence record that doesn't answer the
"non-diabetic" question). Result: `evidence: []`.

```json
{ "evidence": [], "status": "No structured evidence extracted from the retrieved literature.",
  "plan": ["...", "Tool: evidence_extractor", "Evidence Records: 0"] }
```

→ **evidence_validator**.

### 5b. `evidence_validator` — pass 2  →  `corpus_failure`

- **reads:** question, `evidence=[]`, `rerank_top_score=0.29`
- **does:** 1 LLM call. Rewrite already tried, still nothing usable; the specific question
  (metformin → CV mortality in genuinely non-diabetic adults) has essentially no direct RCT
  literature.

```json
{ "evidence_verdict": {
    "sufficient": false, "reason": "corpus_failure",
    "coverage": {"intervention": true, "population": false, "outcome": false},
    "notes": "No PubMed evidence directly addresses CV mortality with metformin in non-diabetic adults; prevention trials show null CV effect in prediabetes." },
  "status": "Evidence insufficient (corpus_failure).",
  "plan": ["...", "Tool: evidence_validator", "Verdict: corpus_failure",
           "Coverage: intervention=True, population=False, outcome=False"] }
```

- **route:** `route_validator` → `reason == "corpus_failure"` → **web_search**.
  *(If `reason` had been `query_failure` again here, the branch is the same because
  `retrieval_attempts(1) < 2` is still true → it would rewrite once more; only after
  `retrieval_attempts` reaches `CRAG_MAX_RETRIES` does an unresolved `query_failure` fall through
  to `web_search`.)*

### 7. `web_search`

- **reads:** `current_query`
- **does:** `tavily_search(query)` → 3 `WebResult`s. (No `TAVILY_API_KEY` → returns `[]` and the
  responder emits the insufficient-evidence message instead.)
- **writes:**

```json
{ "web_results": [
    {"title": "Metformin - StatPearls - NCBI Bookshelf", "url": "https://www.ncbi.nlm.nih.gov/books/NBK518983/",
     "snippet": "Cardiovascular benefit is best established in type 2 diabetes; evidence in non-diabetic populations is limited ..."},
    {"title": "Diabetes Prevention Program Outcomes Study - cardiovascular results", "url": "https://www.nejm.org/...",
     "snippet": "Metformin did not significantly reduce cardiovascular events in adults with prediabetes over long-term follow-up."},
    {"title": "Metformin in patients without diabetes (VA-IMPACT trial overview)", "url": "https://clinicaltrials.gov/study/NCT04264897",
     "snippet": "Ongoing trial testing whether metformin reduces death and CV events in people with prediabetes and established ASCVD."}
  ],
  "used_web_fallback": true,
  "status": "PubMed had no sufficient evidence — pulled 3 web result(s).",
  "plan": ["...", "Tool: web_search (Tavily)", "Web Results: 3"] }
```

- **route:** static edge → **responder**.

### 8. `responder` — web-fallback branch

- **reads:** `used_web_fallback=True` → builds `web_block` from `web_results` and calls
  `build_web_fallback_prompt(history, user_msg, web_block)`.
- **does:** streaming LLM call. `evidence` is empty so `is_grounded_answer` is `False`.
- **writes** (non-grounded path → appends `messages`; `citations: []`):

```json
{ "final_answer": "> ⚠️ This answer is from a general web search, not the curated PubMed evidence base. Verify against primary sources before clinical use.\n\nThere is no strong evidence that metformin reduces cardiovascular mortality in genuinely non-diabetic adults. Its cardiovascular benefit is best established in type 2 diabetes ([Metformin – StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK518983/)). In adults with prediabetes, the Diabetes Prevention Program Outcomes Study found no significant reduction in cardiovascular events over long-term follow-up ([DPPOS CV results](https://www.nejm.org/...)). A dedicated trial in people with prediabetes and established ASCVD ([VA-IMPACT](https://clinicaltrials.gov/study/NCT04264897)) is still ongoing.\n\n### Sources\n- [Metformin – StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK518983/)\n- [DPPOS cardiovascular results](https://www.nejm.org/...)\n- [VA-IMPACT trial](https://clinicaltrials.gov/study/NCT04264897)",
  "citations": [],
  "messages": [{"role": "assistant", "content": "> ⚠️ This answer is from a general web search ..."}],
  "status": "Response generated.",
  "plan": ["..."] }
```

- **route:** `route_responder` → `is_grounded_answer` is `False` (web fallback) → **END**.
  No claim verification for web-sourced answers.

### Final API response

```json
{ "question": "Does metformin reduce cardiovascular mortality in non-diabetic adults?",
  "answer": "> ⚠️ This answer is from a general web search ... VA-IMPACT trial",
  "thought_process": ["Start", "Intent: Clinical Research", "Search Term: metformin cardiovascular mortality non-diabetic",
    "Tool: search_pubmed", "PMIDs Found: 15 (attempt 1)", "Top Relevance: 0.63", "Reranked: top 5 of 10 stored", "Full Text: 3/5",
    "Tool: evidence_extractor", "Evidence Records: 2",
    "Tool: evidence_validator", "Verdict: population_mismatch", "Coverage: intervention=True, population=False, outcome=True",
    "Tool: query_rewriter", "CRAG attempt 1: \"...non-diabetic\" -> \"...prediabetes OR non-diabetic OR insulin resistance...\"",
    "Tool: search_pubmed", "PMIDs Found: 6 (attempt 2)", "Top Relevance: 0.29", "Reranked: top 1 of 1 stored", "Full Text: 0/1",
    "Tool: evidence_extractor", "Evidence Records: 0",
    "Tool: evidence_validator", "Verdict: corpus_failure", "Coverage: intervention=True, population=False, outcome=False",
    "Tool: web_search (Tavily)", "Web Results: 3"],
  "status": "Response generated.",
  "sources": ["... the pass-2 documents (whatever documents holds at END) ..."],
  "citations": [],
  "web_sources": ["... 3 WebResult objects ..."] }
```

The Streamlit UI shows the "outside the curated PubMed evidence base" warning + a **Web sources**
expander (from `web_sources`), and no Citations panel.

---

# Example 3 — "summarise this PMID" (direct path) + one regeneration

**Request**

```json
POST /query
{ "q": "Summarise this paper: https://pubmed.ncbi.nlm.nih.gov/39796530/",
  "thread_id": "demo-3", "pubmed_search_requested": false }
```

### 1. Guardrails — pass (a PubMed URL is on-topic).

### 2. `planner`

- **does:** `_detect_direct_pmid("...pubmed.ncbi.nlm.nih.gov/39796530/...")` → regex match → **no
  LLM call**.
- **writes:**

```json
{ "current_query": "PMID:39796530", "search_params": null, "fresh_search_requested": false,
  "direct_pmid": "39796530",
  "status": "Fetching PMID 39796530...",
  "plan": ["Start", "Intent: Direct PMID Lookup", "PMID: 39796530"] }
```

- **route:** `route_planner` → `current_query != "CONVERSATIONAL"` → **retriever**.

### 3. `retriever` — direct-PMID branch

- **reads:** `direct_pmid="39796530"`
- **does:** `get_pubmed_articles(["39796530"])` → 1 doc; `_attach_full_text` fetches BioC-PMC full
  text (open access here). **No search, no rerank, no session cache.**
- **writes:**

```json
{ "documents": [
    {"pmid": "39796530", "title": "Tang cell counts and COVID-19 disease severity: a prospective cohort",
     "abstract": "In 214 hospitalised patients, lower circulating Tang cell counts at admission correlated with progression to severe COVID-19 (OR 2.8 per SD decrease) ...",
     "journal": "Frontiers in Immunology", "year": "2025", "pub_types": ["Observational Study"],
     "authors": ["Li X", "Sun H"], "has_full_text": true,
     "full_text_markdown": "## Introduction\n...\n## Results\nTang cell counts were inversely associated with ...\n## Discussion\n..."}
  ],
  "retrieval_source": "live_pubmed", "rerank_top_score": 0.0,
  "_background_store_payload": ["... the 1 doc ..."],
  "status": "Found supporting literature.",
  "plan": ["...", "Tool: get_pubmed_articles (direct PMID)", "PMID: 39796530", "Full Text: Yes"] }
```

→ **evidence_extractor**.

### 4. `evidence_extractor`

- **does:** builds the doc block (abstract + full-text markdown, capped at
  `FULLTEXT_EVIDENCE_CHAR_LIMIT`), 1 LLM call.
- **writes:**

```json
{ "evidence": [
    {"pmid": "39796530", "claim": "Lower admission Tang cell counts were associated with progression to severe COVID-19 (OR 2.8 per SD decrease).",
     "evidence": "Lower circulating Tang cell counts at admission correlated with progression to severe COVID-19 (OR 2.8 per SD decrease).",
     "study_type": "Observational Study", "confidence": "moderate"},
    {"pmid": "39796530", "claim": "The association persisted after adjustment for age and comorbidity.",
     "evidence": "Results: Tang cell counts were inversely associated with severity after adjusting for age, sex, and Charlson index.",
     "study_type": "Observational Study", "confidence": "moderate"}
  ],
  "status": "Evidence extracted.",
  "plan": ["...", "Tool: evidence_extractor", "Evidence Records: 2"] }
```

→ **evidence_validator**.

### 5. `evidence_validator` — direct-PMID pass-through

- **does:** 1 LLM call as usual (the node itself doesn't special-case direct PMID). Typically
  returns `sufficient` for a "summarise this paper" task.
- **writes:**

```json
{ "evidence_verdict": {"sufficient": true, "reason": "sufficient",
    "coverage": {"intervention": true, "population": true, "outcome": true},
    "notes": "Single requested paper; evidence records summarise its main finding." },
  "plan": ["...", "Tool: evidence_validator", "Verdict: sufficient", "Coverage: intervention=True, population=True, outcome=True"] }
```

- **route:** `route_validator` → **first check is `if state.get("direct_pmid"): return "responder"`**,
  so a direct-PMID turn goes to the responder *regardless of the verdict* — a failed direct lookup
  is never sent to the web fallback.

### 6. `responder` — pass 1 (grounded)

- **does:** grounded branch, `verify_attempts=0` → no regeneration directive. Streaming LLM call.
- Suppose the draft slightly overstates the design ("**demonstrates that** low Tang cell counts
  **cause** severe COVID-19"):

```json
{ "final_answer": "This prospective cohort of 214 hospitalised patients demonstrates that low circulating Tang cell counts at admission cause progression to severe COVID-19, with roughly 2.8-fold higher odds per standard-deviation decrease [PMID 39796530]. The association held after adjustment for age, sex, and comorbidity burden [PMID 39796530].\n\n### Evidence\n- PMID 39796530 — Observational Study",
  "citations": [
    {"pmid": "39796530", "title": "Tang cell counts and COVID-19 disease severity ...", "journal": "Frontiers in Immunology", "year": "2025", "study_type": "Observational Study", "url": "https://pubmed.ncbi.nlm.nih.gov/39796530/"}
  ],
  "status": "Response generated.",
  "plan": ["..."] }
```

- **route:** `route_responder` → grounded → **claim_verifier**.

### 7. `claim_verifier` — pass 1 → regenerate

- **reads:** `final_answer`, `evidence`, `verify_attempts=0`
- **does:** 1 LLM call → the "cause / demonstrates" wording isn't supported by an **observational**
  study.

```json
{ "needs_regeneration": true,
  "unsupported_claims": ["States low Tang cell counts \"cause\" severe COVID-19 and that the study \"demonstrates\" causation, but the cited evidence is an observational cohort (association only)."],
  "verify_attempts": 1,
  "status": "Regenerating answer — 1 unsupported claim(s).",
  "plan": ["...", "Claim verification: flagged 1 claim(s) — regenerating"] }
```

- **route:** `route_claim_verifier` → `needs_regeneration` **and** `verify_attempts(0) <
  CLAIM_VERIFY_MAX_RETRIES(1)` was true when it decided → **responder**.

### 6b. `responder` — pass 2 (with regeneration directive)

- **reads:** `verify_attempts=1`, `unsupported_claims=[...]` → prepends
  `build_regeneration_directive(unsupported_claims)` to the grounded prompt.
- **writes:** a corrected draft (association language):

```json
{ "final_answer": "This prospective cohort of 214 hospitalised patients found that lower circulating Tang cell counts at admission were **associated** with progression to severe COVID-19 — about 2.8-fold higher odds per standard-deviation decrease [PMID 39796530]. The association persisted after adjustment for age, sex, and comorbidity burden [PMID 39796530]. As an observational study it cannot establish causation.\n\n### Evidence\n- PMID 39796530 — Observational Study",
  "citations": ["... same single Citation ..."],
  "status": "Response generated.",
  "plan": ["..."] }
```

- **route:** still grounded → **claim_verifier**.

### 7b. `claim_verifier` — pass 2 → END

- **reads:** new `final_answer`, `verify_attempts=1`
- **does:** 1 LLM call → `all_claims_supported=true`.
- **writes:**

```json
{ "needs_regeneration": false, "unsupported_claims": [],
  "messages": [{"role": "assistant", "content": "This prospective cohort of 214 hospitalised patients found that lower circulating Tang cell counts ... cannot establish causation.\n\n### Evidence\n- PMID 39796530 — Observational Study"}],
  "status": "Response generated.",
  "plan": ["...", "Claim verification: all claims supported"] }
```

*(If pass 2 had **still** been flagged, `verify_attempts(1) < 1` is false → the `else` branch
runs: append the `### ⚠️ Self-Check Note` block listing the claims to `final_answer` + `messages`,
`plan += ["Claim verification: 1 unsupported claim(s) — caveat attached"]`, then END.)*

- **route:** `route_claim_verifier` → not `needs_regeneration` → **END**.

### Final API response

```json
{ "question": "Summarise this paper: https://pubmed.ncbi.nlm.nih.gov/39796530/",
  "answer": "This prospective cohort of 214 hospitalised patients found that lower circulating Tang cell counts ... cannot establish causation.\n\n### Evidence\n- PMID 39796530 — Observational Study",
  "thought_process": ["Start", "Intent: Direct PMID Lookup", "PMID: 39796530",
    "Tool: get_pubmed_articles (direct PMID)", "PMID: 39796530", "Full Text: Yes",
    "Tool: evidence_extractor", "Evidence Records: 2",
    "Tool: evidence_validator", "Verdict: sufficient", "Coverage: intervention=True, population=True, outcome=True",
    "Claim verification: flagged 1 claim(s) — regenerating",
    "Claim verification: all claims supported"],
  "status": "Response generated.",
  "sources": ["... the 1 PubMedDocument ..."],
  "citations": ["... 1 Citation ..."],
  "web_sources": [] }
```

---

# State-key cheat sheet

| Key | Written by | Read by |
|---|---|---|
| `messages` | entry (`user`), `responder` / `claim_verifier` (`assistant`) | `planner`, `retriever` (turn count), `responder`, history replay |
| `current_query` | `planner`, `query_rewriter` | `retriever`, `evidence_validator`, `web_search`, `responder`, `route_planner` |
| `search_params` | `planner` | `retriever` |
| `direct_pmid` | `planner` | `retriever`, `route_validator` |
| `retrieval_source` | `retriever` | `main.py` (background Qdrant write gate) |
| `retrieval_attempts` | `query_rewriter` | `retriever` (force-fresh), `route_validator` (retry budget) |
| `rerank_top_score` | `retriever` | `evidence_validator` (prompt hint) |
| `documents` | `retriever` | `evidence_extractor`, `responder` (citation join), API `sources` |
| `evidence` | `evidence_extractor` | `evidence_validator`, `responder`, `claim_verifier`, `route_responder` |
| `evidence_verdict` | `evidence_validator` | `route_validator`, `query_rewriter` (notes) |
| `citations` | `responder` | API `citations`, Streamlit |
| `web_results` | `web_search` | `responder` (prompt), API `web_sources`, Streamlit |
| `used_web_fallback` | `web_search` | `responder` (branch), `route_responder` |
| `verify_attempts` | `claim_verifier` | `responder` (directive), `claim_verifier` (budget) |
| `unsupported_claims` | `claim_verifier` | `responder` (`build_regeneration_directive`) |
| `needs_regeneration` | `claim_verifier` | `route_claim_verifier` |
| `plan` | every node (append) | API `thought_process`, Streamlit "Thought process" |
| `final_answer` | `responder`, `claim_verifier` (caveat) | API `answer`, `claim_verifier` (critique input) |
