import streamlit as st
from app_core import handle_user_query

import sys
import os
print("\n".join(sys.path))
sys.path.insert(0, os.path.abspath("."))

st.set_page_config(page_title="Omni Assistant", layout="wide")
st.title("🤖 Omni Assistant")
st.markdown("Ask a question based on internal documentation, Slack messages, or metrics.")

query = st.text_input("Your question:")

if query:
    with st.spinner("Analyzing and searching..."):
        answer, docs = handle_user_query(query)

    st.markdown("## ✅ Answer")
    st.markdown(answer)

    if docs:
        st.markdown("---")
        st.markdown("## 📄 Source Highlights")
        for doc in docs:
            st.markdown(f"**{doc['title']}** ({doc['source']})")
            if doc["url"]:
                st.markdown(f"[🔗 Source Link]({doc['url']})")
            st.markdown(f"> {doc['content'][:500]}{'...' if len(doc['content']) > 500 else ''}")
            st.markdown("---")
