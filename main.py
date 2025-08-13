# main.py (your Streamlit app)
import streamlit as st
from planning.catalog import tool_catalog
from app_modes import run_query

st.set_page_config(page_title="Omni-GPT", layout="centered")
st.title("Omni-GPT")

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI controls
mode = st.radio("Mode", ["search", "planned"], index=0, horizontal=True)
# Show only human-facing names if you want; here we use IDs
tool_ids = list(tool_catalog.keys())
selected_tools = st.multiselect("Tools", tool_ids, default=[tid for tid in tool_ids if tid != "typesense_search"])

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about Omni...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, steps, trace, docs = run_query(mode, user_input, selected_tools)

        st.markdown(f"**Response:**\n\n{answer}")

        if mode == "planned":
            with st.expander("Plan & Trace"):
                st.json({"steps": steps, "trace": trace})

        if docs:
            with st.expander("Sources"):
                for d in docs:
                    link = f"[🔗]({d['url']})" if d.get("url") else ""
                    st.markdown(f"- **{d['title']}** {link}")

    st.session_state.messages.append({"role": "assistant", "content": f"**Answer:**\n\n{answer}"})
