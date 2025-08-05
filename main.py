import streamlit as st
from app_core import handle_user_query

import sys, os
sys.path.insert(0, os.path.abspath("."))

st.set_page_config(page_title="Omni Assistant", layout="wide")

st.title("🤖 Omni Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (anchored at bottom)
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing and searching..."):
            answer, docs = handle_user_query(user_input)

        # Format answer with optional sources
        full_answer = f"**Answer:**\n\n{answer}"
        if docs:
            full_answer += "\n\n---\n**Sources:**\n"
            for doc in docs:
                doc_link = f"[🔗]({doc['url']})" if doc["url"] else ""
                full_answer += f"- **{doc['title']}** {doc_link}\n"


        st.markdown(full_answer)
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
