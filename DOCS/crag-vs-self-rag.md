# Is this system CRAG or Self‑RAG?

Judging by mechanism, not labels: **this is CRAG at its core, with one Self‑RAG-style
mechanism grafted on (post-generation grounding critique + regenerate).** It is not Self‑RAG.

## Why it's CRAG

The defining CRAG control flow is: *retrieve → a lightweight evaluator grades the retrieval →
branch into {use it | correct it | replace it with web search} → generate.* That is exactly the
spine here:

| CRAG mechanism | This system |
|---|---|
| Retrieval evaluator emits a graded verdict (Correct / Ambiguous / Incorrect) | `evidence_validator` emits `sufficient` / `query_failure` \| `population_mismatch` / `corpus_failure` — the same trichotomy, generalized |
| Knowledge refinement: decompose retrieved docs into filtered "knowledge strips" | `evidence_extractor` decomposes abstracts into filtered, PMID-attributed claim records |
| **Incorrect → discard retrieval, do a web search** | **`corpus_failure` → Tavily web search** (with disclaimer). This is CRAG's signature move and it's here almost verbatim |
| Query rewriting to drive corrective re-retrieval | `query_rewriter` rewrites the PubMed query and re-runs retrieval |

If anything it's *more* corrective than vanilla CRAG: vanilla CRAG evaluates once and feeds
forward, whereas this has a real bounded loop (`validator → query_rewriter → retriever →
validator …`), and the evaluator judges *answerability/coverage* ("diabetic vs. non-diabetic
population mismatch"), not just raw document relevance.

## The Self‑RAG borrowing

`claim_verifier` is a Self‑RAG idea: it does an `IsSupported`-style check — "are the generated
answer's claims grounded in the retrieved evidence?" — and loops back to regenerate if not. The
planner's retrieve-vs-answer-from-memory gate is also Self‑RAG-flavored adaptive retrieval
(`Retrieve` token), though that's now a generic pattern.

## Why it is *not* Self‑RAG

Self‑RAG's actual architecture is absent:

- **No reflection-token policy** — Self‑RAG is *one model* trained to emit
  `Retrieve`/`IsRel`/`IsSup`/`IsUse` tokens inline. Here these are separate nodes / separate LLM
  calls.
- **No interleaved on-demand retrieval during decoding** — Self‑RAG can retrieve per segment,
  mid-generation, repeatedly. Here retrieval is a discrete upstream phase that finishes before
  generation starts.
- **No parallel-candidate generation + segment-level beam search** over critique scores. The
  `claim_verifier` regeneration is a single sequential bounded retry
  (`CLAIM_VERIFY_MAX_RETRIES = 1`), not "generate K candidates from K passages and select by
  weighted `IsRel`/`IsSup`/`IsUse`."
- **No `IsUse` usefulness scoring** driving output selection.

## Bottom line

Accurate description: **Corrective RAG (CRAG) with a self-reflective output-grounding check.**
The retrieval-evaluation-and-correction loop plus web-search fallback is pure CRAG; the only
Self‑RAG element is verifying the *generation* against evidence and regenerating. And because all
routing is deterministic graph edges with bounded loops (no LLM choosing tools), it's
*structured* corrective RAG, not agentic RAG — which is consistent with the arch-feedback doc's
thesis that sufficiency is control flow, not an agent decision.

## References

- Yan et al., 2024 — *Corrective Retrieval Augmented Generation* (CRAG)
- Asai et al., 2023 — *Self-RAG: Learning to Retrieve, Generate, and Critique through
  Self-Reflection*
- `DOCS/arch-feedback.md` — the architecture critique this refactor implements
