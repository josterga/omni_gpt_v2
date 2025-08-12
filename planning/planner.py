import json
from typing import List
from .types import ToolCatalog, PlanStep

PLANNER_PROMPT = """
You are a query planner for an agent with multiple tools.
Decompose the user question into a sequence of tool calls whose combined outputs should answer the question.

Rules:
- Select only tools that materially contribute to the question.
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
            f"- {spec['name']}: {spec['description']}" for spec in tool_catalog.values()
        )
        prompt = f"""{PLANNER_PROMPT}

Available tools:
{tool_descriptions}

User question:
{user_text}
"""
        llm, cfg = self.llm_factory(provider="openai", model="gpt-4o-mini")
        raw = llm.chat([{"role": "user", "content": prompt}],
                       model=cfg["model"], **cfg.get("params", {"temperature": 0}))
        txt = _strip_code_fences(str(raw))
        steps = json.loads(txt)
        assert isinstance(steps, list)
        for s in steps:
            assert "id" in s and "tool" in s and "args" in s
        return steps
