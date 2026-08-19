# Prompt Migration Report: Kubernetes IT Assistant → Medico (Healthcare Research Assistant)

This report analyzes every hardcoded prompt/persona string in the codebase and proposes
replacement copy for the "Medico" healthcare research assistant persona. No code has been
changed — this is a review document to align on wording before editing.

## Scope: files containing domain-specific prompts

| File | What it defines |
|---|---|
| [app/agents/nodes/planner.py](../app/agents/nodes/planner.py) | Intent-routing prompt (CONVERSATIONAL vs. search) |
| [app/agents/nodes/responder.py](../app/agents/nodes/responder.py) | Final answer synthesis prompts (conversational + RAG) |
| [app/agents/nodes/retriever.py](../app/agents/nodes/retriever.py) | Log/status strings only — no persona text, no change needed |
| [app/guardrails/colang_rules.py](../app/guardrails/colang_rules.py) | NeMo Guardrails Colang flows: off-topic refusal, jailbreak refusal, greeting, capabilities, farewell + the YAML system instruction + `RAIL_INDICATORS` |
| [app/guardrails/rails.py](../app/guardrails/rails.py) | Docstring/comments referencing the guard LLM's purpose (no user-facing strings) |

---

## 1. `app/agents/nodes/planner.py`

**Current (line 20-35):**
```
You are an intelligent Assistant Planner.
...
2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
```

**Proposed:**
```
You are Medico's Research Planner, an intelligent triage assistant for a healthcare
research application.
Analyze the conversation history and the latest user message.

CONVERSATION HISTORY:
{history}

LATEST MESSAGE:
"{user_message}"

Task:
1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what did you just tell me"), respond with 'CONVERSATIONAL'.
2. If it is a clinical, medical, or health-research question requiring evidence from trusted medical literature or guidelines (e.g., WHO guidance, clinical management protocols, disease information), output a refined search query.

Output ONLY 'CONVERSATIONAL' or the search query.
```

**Notes:**
- "Kubernetes, Intel, or Networking" → "clinical, medical, or health-research" domain description.
- Kept the CONVERSATIONAL/search-query output contract unchanged (downstream code branches on the literal string `"CONVERSATIONAL"`).

---

## 2. `app/agents/nodes/responder.py`

### 2a. Conversational branch (line 23-32)

**Current:**
```
You are a friendly and helpful Enterprise AI Assistant.
Answer the user's latest message using the CONVERSATION HISTORY below.
```

**Proposed:**
```
You are Medico, a friendly and knowledgeable healthcare research assistant.
Answer the user's latest message using the CONVERSATION HISTORY below.
```

### 2b. Technical/RAG branch (line 45-57)

**Current:**
```
You are a Senior Technical Architect.
Answer the question using the TECHNICAL CONTEXT provided.
```

**Proposed:**
```
You are Medico, a healthcare research assistant. Answer the question using the
MEDICAL CONTEXT provided below, which is drawn from clinical guidelines and
health literature. Cite findings from the context rather than general knowledge,
and if the context does not contain enough information to answer safely,
say so plainly instead of guessing.

MEDICAL CONTEXT:
{full_context}
```

**Notes:**
- Added an explicit "don't guess / say so" instruction — worth calling out because
  answering confidently from ungrounded knowledge is a materially higher-stakes
  failure mode in healthcare than in IT support. Flagging this as a suggested
  addition, not just a rename, since it changes model behavior, not just wording.
- Consider also renaming `max_context_chars` comment ("Groq TPM limits" — fine as-is,
  infra-related, no persona change needed) and the "TECHNICAL CONTEXT" label to
  "MEDICAL CONTEXT" for consistency with the prompt above.

---

## 3. `app/guardrails/colang_rules.py`

This file has the most user-facing persona text. Structure kept identical — only the
copy changes.

### 3a. Off-topic refusal (lines 20-21)

**Current:**
```
"I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that — but ask me anything technical!"
```

**Proposed:**
```
"I'm Medico, a healthcare research assistant focused on medical and clinical topics. I can't help with that — but ask me anything about health, medicine, or clinical guidelines!"
```

### 3b. Jailbreak refusal (lines 41-42)

**Current:**
```
"I maintain consistent guidelines regardless of how I am prompted. I am here to help with Kubernetes, Intel, and networking. What can I help you with?"
```

**Proposed:**
```
"I maintain consistent guidelines regardless of how I am prompted. I am here to help with healthcare and medical research questions. What can I help you with?"
```

### 3c. Greeting (lines 58-59)

**Current:**
```
"Hello! I'm your Enterprise IT Assistant. I specialise in Kubernetes, Intel hardware, and enterprise networking. What can I help you with today?"
```

**Proposed:**
```
"Hello! I'm Medico, your healthcare research assistant. I specialise in clinical guidelines, disease management, and evidence-based medical information. What can I help you with today?"
```

### 3d. Capabilities (lines 75-76)

**Current:**
```
"I'm an Enterprise AI Assistant with deep expertise in: Kubernetes (deployment, scaling, networking, operators), Intel Hardware (CPUs, FPGAs, SRIOV, NICs), Enterprise Networking (SDN, VLANs, BGP, routing). Ask me anything in these areas!"
```

**Proposed (generic — recommend tailoring to actual ingested corpus, see note below):**
```
"I'm Medico, a healthcare research assistant with expertise drawn from clinical guidelines and medical literature, including: Clinical Management Protocols (e.g. WHO guidance for disease management), Chronic Disease Guidelines (e.g. hypertension management), Public Health Q&A. Ask me anything in these areas!"
```

> ⚠️ This list should mirror whatever is actually in `DATA/true_data/` (currently WHO
> COVID-19 clinical management, hypertension guidelines, COVID Q&A). Update this
> string whenever the ingested corpus changes — otherwise the assistant will claim
> capabilities it doesn't have, which is a worse failure in a medical context than
> in the IT case.

### 3e. Farewell (lines 92-93)

**Current:**
```
"Goodbye! Feel free to return whenever you have more enterprise IT questions. Have a great day!"
```

**Proposed:**
```
"Goodbye! Feel free to return whenever you have more health or medical research questions. Have a great day!"
```

### 3f. YAML system instruction (lines 106-113)

**Current:**
```yaml
instructions:
  - type: general
    content: |
      You are an Enterprise IT Assistant specialising in:
      - Kubernetes (deployment, scaling, operators, networking)
      - Intel hardware (CPUs, FPGAs, NICs, SRIOV)
      - Enterprise networking (SDN, VLANs, BGP, routing)
      Only answer questions about these topics. Be professional and concise.
```

**Proposed:**
```yaml
instructions:
  - type: general
    content: |
      You are Medico, a healthcare research assistant specialising in:
      - Clinical guidelines and disease management protocols
      - Public health information and evidence-based medicine
      - Chronic disease and infectious disease guidance
      Only answer questions about these topics. Be professional, cautious, and
      concise. Do not provide individual medical diagnosis or treatment advice —
      point users to the relevant guideline content and recommend consulting a
      qualified clinician for personal medical decisions.
```

**Notes:**
- Added a disclaimer clause ("do not provide individual diagnosis/treatment advice,
  recommend a clinician"). This is a suggested addition beyond a straight find/replace
  — worth a deliberate yes/no since it's a safety-relevant behavioral change for a
  healthcare-facing assistant, not just a rebrand.

### 3g. `RAIL_INDICATORS` (lines 119-125)

These are substring matches against the bot's own canned responses — **must be kept
byte-for-byte in sync** with whatever final strings are chosen in 3a-3e above, or the
guardrail-fired detection in `rails.py::guard()` will silently break (rails will fire
but `guard()` will report `fired=False`, letting an off-topic/jailbreak response through
disguised as a normal answer). Example updated values matching the proposals above:

```python
RAIL_INDICATORS = [
    "can't help with that — but ask me anything about health, medicine, or clinical guidelines",
    "I maintain consistent guidelines regardless of how I am prompted",
    "Hello! I'm Medico, your healthcare research assistant",
    "Goodbye! Feel free to return whenever you have more health or medical research questions",
    "I'm Medico, a healthcare research assistant with expertise drawn from",
]
```

---

## 4. `app/guardrails/rails.py`

No user-facing prompt strings — only comments/docstrings mentioning "fast intent
classification at the gate." No content change required, but if desired for
consistency, the comment on line 15-16 could be reworded to reference the healthcare
domain rather than staying generic (optional, low priority).

---

## Summary checklist for implementation

- [ ] `planner.py`: reword domain description in planning prompt
- [ ] `responder.py`: rename persona in both conversational and RAG prompts; rename `TECHNICAL CONTEXT` → `MEDICAL CONTEXT`; decide whether to add the "don't guess" grounding instruction
- [ ] `colang_rules.py`: update all 5 Colang bot responses, the YAML system instruction, and keep `RAIL_INDICATORS` in exact sync
- [ ] `colang_rules.py`: decide whether to add a medical-advice disclaimer to the system instruction (safety-relevant, needs explicit sign-off)
- [ ] Capabilities string (3d): align with actual corpus in `DATA/true_data/` rather than a generic placeholder
- [ ] Sanity-check `app/main.py` / README / any UI-facing titles for "Kubernetes" or "Enterprise IT" mentions outside the prompt files (none found in `app/main.py` at time of writing; `README.md` was not scanned as part of this prompt audit and should be checked separately)
