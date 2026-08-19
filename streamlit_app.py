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

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for i, doc in enumerate(turn["sources"], start=1):
                    st.markdown(f"**{i}.** {doc[:500]}{'…' if len(doc) > 500 else ''}")
        if turn.get("thought_process"):
            with st.expander("Thought process"):
                for step in turn["thought_process"]:
                    st.markdown(f"- {step}")

if question := st.chat_input("Ask about clinical guidelines, e.g. WHO hypertension management…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"q": question, "thread_id": st.session_state.thread_id},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e:
                st.error(f"Could not reach the backend at {BACKEND_URL}: {e}")
                st.stop()

        answer = data.get("answer", "Sorry, I couldn't generate a response.")
        sources = data.get("sources") or []
        thought_process = data.get("thought_process") or []

        st.markdown(answer)
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, doc in enumerate(sources, start=1):
                    st.markdown(f"**{i}.** {doc[:500]}{'…' if len(doc) > 500 else ''}")
        if thought_process:
            with st.expander("Thought process"):
                for step in thought_process:
                    st.markdown(f"- {step}")

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "thought_process": thought_process,
    })
