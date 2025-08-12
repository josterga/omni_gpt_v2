from typing import Any, Dict, List, Literal, TypedDict, Callable

class ToolResult(TypedDict, total=False):
    kind: Literal["text", "json", "docs", "list"]
    value: Any
    preview: str
    metadata: Dict[str, Any]

class ToolSpec(TypedDict, total=False):
    name: str
    description: str
    produces: Literal["text", "json", "docs", "list"]
    run: Callable[[Dict[str, Any]], ToolResult]
    args_schema: Dict[str, Any]
    parallelizable: bool

ToolCatalog = Dict[str, ToolSpec]

# Planner step shape
class PlanStep(TypedDict, total=False):
    id: str
    tool: str
    args: Dict[str, Any]
    parallel_group: str  # optional
