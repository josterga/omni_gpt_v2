import streamlit as st
from app_core import handle_user_query

import sys, os
sys.path.insert(0, os.path.abspath("."))

st.set_page_config(page_title="Omni Assistant", layout="wide")

st.title("Omni-GPT")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (anchored at bottom)
user_input = st.chat_input("Ask anything about Omni...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, docs = handle_user_query(user_input)

        # Format answer with optional sources
        st.markdown(f"**Response:**\n\n{answer}")

        if docs:
            with st.expander("Sources"):
                for doc in docs:
                    doc_link = f"[🔗]({doc['url']})" if doc["url"] else ""
                    st.markdown(f"- **{doc['title']}** {doc_link}")
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Answer:**\n\n{answer}"
        })
