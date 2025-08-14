import json
from typing import List
from .types import ToolCatalog, PlanStep

PLANNER_PROMPT = """
You are a query planner for an agent with multiple tools.
Decompose the user question into a sequence of tool calls whose combined outputs should answer the question.

Rules:
- Select only tools that materially contribute to the question.
- Use only the tool IDs exactly as given in the list below when specifying which tool to use.
- Tool descriptions will contain detail about what data they can provide.
- If a step depends on an earlier step's output, reference it with: {"$ref": "stepID.output[<jsonpath>]"}.
- Keep arguments concrete and minimal.
- Return ONLY a flat JSON array of steps. No commentary.

Step shape:
{
  "id": "step1",
  "tool": "<tool_name>",
  "args": { ... },
  "parallel_group": "A"  # optional
}
""".strip()

def _strip_code_fences(txt: str) -> str:
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

class ToolPlanner:
    def __init__(self, llm_factory):
        self.llm_factory = llm_factory  # your get_llm

    def plan(self, user_text: str, tool_catalog: ToolCatalog) -> List[PlanStep]:
        tool_descriptions = "\n".join(
            f"- {tool_id} ({spec['name']}): {spec['description']}"
            for tool_id, spec in tool_catalog.items()
        )
        prompt = f"""{PLANNER_PROMPT}

Available tools:
{tool_descriptions}

User question:
{user_text}
"""
        
        try:
            llm, cfg = self.llm_factory(provider="openai", model="gpt-4o-mini")
            
            # Check if LLM is available
            if llm is None:
                print("⚠️  LLM not available, using fallback planning")
                return self._fallback_plan(user_text, tool_catalog)
            
            # Try to use the LLM for planning
            raw = llm.chat([{"role": "user", "content": prompt}],
                           model=cfg["model"], **cfg.get("params", {"temperature": 0}))
            txt = _strip_code_fences(str(raw))
            steps = json.loads(txt)
            assert isinstance(steps, list)
            for s in steps:
                assert "id" in s and "tool" in s and "args" in s
                if s["tool"] not in tool_catalog:
                    raise ValueError(f"Planner selected unknown tool: {s['tool']}")
            return steps
            
        except Exception as e:
            print(f"⚠️  LLM planning failed: {e}, using fallback planning")
            return self._fallback_plan(user_text, tool_catalog)
    
    def _fallback_plan(self, user_text: str, tool_catalog: ToolCatalog) -> List[PlanStep]:
        """Fallback planning when LLM is not available."""
        # Simple fallback: use the first available tool with basic args
        available_tools = list(tool_catalog.keys())
        if not available_tools:
            return []
        
        # Create a simple plan with the first available tool
        return [{
            "id": "fallback_step",
            "tool": available_tools[0],
            "args": {"query": user_text}
        }]
