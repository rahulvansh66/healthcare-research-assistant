import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Medico — Healthcare Research Assistant", page_icon="🩺")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🩺 Medico — Healthcare Research Assistant")
st.caption("Grounded in WHO clinical guidelines. Not a substitute for professional medical advice.")

with st.sidebar:
    st.subheader("Session")
    st.text(f"Thread: {st.session_state.thread_id[:8]}")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption(f"Backend: {BACKEND_URL}")

def _render_citations(citations):
    with st.expander(f"Citations ({len(citations)})"):
        for i, c in enumerate(citations, start=1):
            year = f" ({c['year']})" if c.get("year") else ""
            st.markdown(
                f"**{i}.** [{c.get('title', 'Untitled')}]({c.get('url', '#')}) — "
                f"{c.get('journal', '')}{year} · {c.get('study_type', 'Unknown')} · PMID: {c.get('pmid', '')}"
            )


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("citations"):
            _render_citations(turn["citations"])
        if turn.get("thought_process"):
            with st.expander("Thought process"):
                for step in turn["thought_process"]:
                    st.markdown(f"- {step}")

pubmed_search = st.toggle(
    "PubMed Search",
    key="pubmed_search_toggle",
    help="Force a fresh PubMed search instead of reusing this conversation's cached results.",
)

if question := st.chat_input("Ask about clinical guidelines, e.g. WHO hypertension management…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "q": question,
                        "thread_id": st.session_state.thread_id,
                        "pubmed_search_requested": pubmed_search,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e:
                st.error(f"Could not reach the backend at {BACKEND_URL}: {e}")
                st.stop()

        answer = data.get("answer", "Sorry, I couldn't generate a response.")
        citations = data.get("citations") or []
        thought_process = data.get("thought_process") or []

        st.markdown(answer)
        if citations:
            _render_citations(citations)
        if thought_process:
            with st.expander("Thought process"):
                for step in thought_process:
                    st.markdown(f"- {step}")

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "thought_process": thought_process,
    })
