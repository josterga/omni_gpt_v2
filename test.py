# tests/test_planner_exec_real_tools.py
import json
from planning.catalog import tool_catalog
from planning.planner import ToolPlanner
from planning.executor import ToolExecutor
from applied_ai.generation.generation.router import get_llm

def test_planner_with_real_tools():
    user_question = "what are the most recent 3 meetings today?"

    planner = ToolPlanner(get_llm)
    steps = planner.plan(user_question, tool_catalog)
    assert isinstance(steps, list) and len(steps) >= 1
    
    print("\n=== Planned Execution Summary ===")
    for step in steps:
        print(f"[{step['id']}] {step['tool']}")
        print(f"Args: {json.dumps(step['args'], indent=2)}")
        print("-" * 40)
    print("=================================\n")

    executor = ToolExecutor(tool_catalog)
    trace, evidence = executor.run(steps)

    print("\n--- Planned Steps ---")
    print(json.dumps(steps, indent=2))
    print("\n--- Execution Trace ---")
    print(json.dumps(trace, indent=2))
    print("\n--- Evidence ---")
    print(json.dumps(evidence, indent=2))

    assert len(evidence) > 0, "No evidence returned from real tools"
    print(json.dumps(trace, indent=2))

if __name__ == "__main__":
    test_planner_with_real_tools()
    print("\nPlanner executed real tools successfully.")
