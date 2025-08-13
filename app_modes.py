# app_modes.py
from typing import List, Tuple, Dict, Any
from orchestrators import planned as planned_mode
from orchestrators import direct as direct_mode

def run_query(mode: str, query: str, allowed_tools: List[str]):
    if mode == "planned":
        return planned_mode.run(query, allowed_tool_ids=allowed_tools)
    elif mode == "search":
        return direct_mode.run(query, allowed_tool_ids=allowed_tools)
    raise ValueError(f"Unknown mode: {mode}")
