"""
UI Components for tool selection and organization.
Provides sidebar-based tool selection with categories and descriptions.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from planning.catalog import get_tools_by_category, get_tool_display_info


def render_tool_sidebar(
    selected_tools: Optional[List[str]] = None,
    key_prefix: str = "tool_selector"
) -> List[str]:
    """
    Renders a sidebar with categorized tool selection.
    
    Args:
        selected_tools: List of pre-selected tool IDs
        key_prefix: Unique prefix for Streamlit widget keys
        
    Returns:
        List of selected tool IDs
    """
    if selected_tools is None:
        selected_tools = []
    
    st.sidebar.header("Tool Selection")
    
    # Get categorized tools
    categorized_tools = get_tools_by_category()
    tool_display_info = get_tool_display_info()
    
    selected = []
    
    # Render each category
    for category, tools in categorized_tools.items():
        if not tools:
            continue
            
        st.sidebar.subheader(f"{category}")
        
        for tool_id, tool in tools.items():
            display_info = tool_display_info.get(tool_id, {})
            
            # Create unique key for each checkbox
            checkbox_key = f"{key_prefix}_{tool_id}"
            
            # Check if tool is selected
            is_selected = tool_id in selected_tools
            
            # Create checkbox with tool info
            if st.sidebar.checkbox(
                f"{display_info.get('name', tool_id)}",
                value=is_selected,
                key=checkbox_key,
                help=display_info.get('description', '')
            ):
                selected.append(tool_id)
    
    return selected


def get_tool_selection_widget(
    default_selection: Optional[List[str]] = None,
    key_prefix: str = "main_tool_selector"
) -> List[str]:
    """
    Renders only the sidebar tool selection (no main area display).
    
    Args:
        default_selection: List of tool IDs to pre-select
        key_prefix: Unique prefix for Streamlit widget keys
        
    Returns:
        List of selected tool IDs
    """
    return render_tool_sidebar(
        selected_tools=default_selection,
        key_prefix=key_prefix
    )


# Example usage in your main app:
"""
# In your main.py or wherever you want the tool selection UI:

from tooling.ui_components import get_tool_selection_widget

# Render tool selection UI (sidebar only)
selected_tools = get_tool_selection_widget(
    default_selection=["slack_search", "docs_embed_search"],
    key_prefix="main"
)

# Use selected_tools in your orchestrator
if selected_tools:
    answer, steps, trace, docs = run(query, allowed_tool_ids=selected_tools)
""" 