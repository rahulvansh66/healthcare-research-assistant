import json
import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Medico — Clinical Research Assistant", page_icon="🩺")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🩺 Medico — Clinical Research Assistant")
# st.caption("Grounded in WHO clinical guidelines. Not a substitute for professional medical advice.")

def _fetch_sessions():
    try:
        resp = requests.get(f"{BACKEND_URL}/sessions", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


def _load_session(thread_id: str):
    """Switches to an existing thread, replaying its messages from the
    backend. Citations/thought-process aren't stored per historical turn,
    so replayed assistant turns show text only."""
    st.session_state.thread_id = thread_id
    try:
        resp = requests.get(f"{BACKEND_URL}/sessions/{thread_id}/history", timeout=10)
        resp.raise_for_status()
        messages = resp.json().get("messages", [])
    except requests.RequestException:
        messages = []
    st.session_state.history = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    st.rerun()


with st.sidebar:
    st.subheader("Session")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.divider()
    st.caption("Recent")
    for session in _fetch_sessions():
        is_current = session["thread_id"] == st.session_state.thread_id
        label = session["title"] or "(untitled)"
        if st.button(
            label,
            key=f"session_{session['thread_id']}",
            type="primary" if is_current else "secondary",
            use_container_width=True,
        ):
            if not is_current:
                _load_session(session["thread_id"])
    st.divider()
    st.caption(f"Backend: {BACKEND_URL}")

def _stream_tokens(response, result_holder):
    """Yields answer text chunks for st.write_stream; stashes the trailing
    'done'/'error' NDJSON line in result_holder since write_stream only wants strings."""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        event = json.loads(line)
        if event["type"] == "token":
            yield event["content"]
        elif event["type"] == "done":
            result_holder["done"] = event
        elif event["type"] == "error":
            result_holder["error"] = event["message"]


def _render_citations(citations):
    with st.expander(f"Citations ({len(citations)})"):
        for i, c in enumerate(citations, start=1):
            year = f" ({c['year']})" if c.get("year") else ""
            st.markdown(
                f"**{i}.** [{c.get('title', 'Untitled')}]({c.get('url', '#')}) — "
                f"{c.get('journal', '')}{year} · {c.get('study_type', 'Unknown')} · PMID: {c.get('pmid', '')}"
            )


def _render_web_sources(web_sources):
    st.warning(
        "Answered from a general web search — outside the curated PubMed evidence base. "
        "Verify against primary sources before clinical use."
    )
    with st.expander(f"Web sources ({len(web_sources)})"):
        for i, s in enumerate(web_sources, start=1):
            st.markdown(f"**{i}.** [{s.get('title', 'Untitled')}]({s.get('url', '#')})")


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("web_sources"):
            _render_web_sources(turn["web_sources"])
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
        try:
            response = requests.post(
                f"{BACKEND_URL}/query/stream",
                json={
                    "q": question,
                    "thread_id": st.session_state.thread_id,
                    "pubmed_search_requested": pubmed_search,
                },
                timeout=60,
                stream=True,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {e}")
            st.stop()

        result_holder = {}
        answer = st.write_stream(_stream_tokens(response, result_holder))

        if result_holder.get("error"):
            answer = result_holder["error"]
            st.markdown(answer)
            citations = []
            thought_process = []
            web_sources = []
        else:
            done = result_holder.get("done", {})
            citations = done.get("citations") or []
            thought_process = done.get("thought_process") or []
            web_sources = done.get("web_sources") or []
            if web_sources:
                _render_web_sources(web_sources)
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
        "web_sources": web_sources,
    })
