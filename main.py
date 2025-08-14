# main.py (your Streamlit app)
import streamlit as st

st.set_page_config(page_title="Omni-GPT", layout="wide")  # Wide layout for sidebar
st.title("Omni-GPT")

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_tools" not in st.session_state:
    st.session_state.selected_tools = []
if "lite_mode" not in st.session_state:
    st.session_state.lite_mode = False

def get_tool_catalog():
    """Get tool catalog - imported here to avoid circular imports."""
    try:
        from planning.catalog import tool_catalog
        return tool_catalog
    except ImportError as e:
        st.error(f"Failed to import tool catalog: {e}")
        return {}

def get_run_query():
    """Get run_query function - imported here to avoid circular imports."""
    try:
        from app_modes import run_query
        return run_query
    except ImportError as e:
        st.error(f"Import failed: {e}")
        return lambda mode, query, tools: ("Error: Import failed", [], {}, [])

# Get tool catalog and run_query function
tool_catalog = get_tool_catalog()
run_query = get_run_query()

# Default tools for each mode
default_tools = {
    "search": ["slack_search", "docs_embed_search", "community_embed_search"],
    "planned": ["slack_search", "docs_embed_search", "community_embed_search", "mcp_query", "fathom_list_meetings"]
}

# Lite Mode toggle - automatically configures typesense_search + search mode
# This needs to be before mode selection to properly override
col1, col2 = st.columns([3, 1])
with col1:
    st.write("")  # Spacer for alignment
with col2:
    lite_mode = st.toggle(
        "🚀 Lite Mode", 
        value=st.session_state.lite_mode,
        help="Quick setup: Typesense search only. Toggle off to return to normal mode selection."
    )
    
    # Handle lite mode toggle changes
    if lite_mode != st.session_state.lite_mode:
        st.session_state.lite_mode = lite_mode
        if lite_mode:
            # Lite mode ON: set to search + typesense
            st.session_state.current_mode = "search"
            st.session_state.selected_tools = ["typesense_search"]
        else:
            # Lite mode OFF: reset to default for current mode
            current_mode = st.session_state.get("current_mode", "search")
            st.session_state.selected_tools = default_tools.get(current_mode, [])
        st.rerun()

# Mode selection (after lite mode toggle to allow override)
if st.session_state.lite_mode:
    # Lite mode is ON: force search mode
    mode = "search"
else:
    # Lite mode is OFF: allow normal mode selection
    mode = st.radio(
        "Mode", 
        ["search", "planned"], 
        index=0, 
        horizontal=True,
        help=(
            "**Search:**\n"
            "- Direct search across selected sources.\n"
            "- Note: Fathom is only supported in Planned mode. MCP in Search mode is only invoked by metric queries (how many, total, etc.) \n"
             "**Planned:**\n"
            "- LLM-orchestrated multi-tool search.\n"
            "- Uses gpt-5 with increased context window\n\n"
        )
    )

# Update default selection when mode changes (only when not in lite mode)
if not st.session_state.lite_mode and ("current_mode" not in st.session_state or st.session_state.current_mode != mode):
    st.session_state.current_mode = mode
    st.session_state.selected_tools = default_tools.get(mode, [])

# Sidebar tool selection using the new UI components
try:
    from tooling.ui_components import get_tool_selection_widget
    
    # Get selected tools from sidebar
    selected_tools = get_tool_selection_widget(
        default_selection=st.session_state.selected_tools,
        key_prefix="main"
    )
    
    # Update session state with current selection
    st.session_state.selected_tools = selected_tools
    
except ImportError as e:
    st.sidebar.error(f"Failed to load tool selection UI: {e}")
    # Fallback to simple multiselect
    tool_ids = list(tool_catalog.keys()) if tool_catalog else []
    selected_tools = st.sidebar.multiselect(
        "Tools (Fallback)", 
        tool_ids, 
        default=st.session_state.selected_tools
    )
    st.session_state.selected_tools = selected_tools

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

