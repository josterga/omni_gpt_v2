# app_core.py
# Plan → Execute → Synthesize core orchestrator.
# - Uses your ToolPlanner + ToolExecutor + tool_catalog
# - Keeps the synthesizer in this file (as requested)
# - Normalizes mixed tool evidence into RAG-ready docs for the synth

import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv

# Planner/executor & tools
from planning.catalog import tool_catalog
from planning.planner import ToolPlanner
from planning.executor import ToolExecutor

# Your LLM router (monorepo generation module)
from applied_ai.generation.generation.router import get_llm  # get_llm entrypoint:contentReference[oaicite:2]{index=2}

# ---------------------------
# Config / constants
# ---------------------------

load_dotenv()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Evidence shaping knobs
DEFAULT_JSON_SNIPPET_LEN = int(os.getenv("PLANNER_JSON_SNIPPET_LEN", "800"))
DEFAULT_TEXT_SNIPPET_LEN = int(os.getenv("PLANNER_TEXT_SNIPPET_LEN", "1200"))
MAX_DOCS_FOR_SYNTH = int(os.getenv("PLANNER_MAX_DOCS_FOR_SYNTH", "30"))

# ---------------------------
# Synthesizer (lives here)
# ---------------------------

def synthesize_answer(
    query: str,
    docs: List[Dict[str, Any]],
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Very lightweight synthesizer:
    - Consumes a list of {title, url, content, source}
    - Uses get_llm(...) to produce an answer using ONLY provided context
    """
    llm, cfg = get_llm(provider=provider, model=model)  # generation.get_llm:contentReference[oaicite:3]{index=3}

    # Compact context assembly
    lines = []
    for d in docs:
        title = d.get("title", "")
        url = d.get("url", "")
        content = d.get("content", "")
        src = d.get("source", "")
        lines.append(f"{src} | {title} ({url}):\n{content}")
    context = "\n\n".join(lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for Omni Analytics. "
                "Use ONLY the provided context. If the context is insufficient, "
                "state exactly what is missing."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User Question:\n{query}\n\n"
                f"Available Information:\n{context}\n\n"
                "Write a concise answer using only the information above. "
                "Reference the sources by name when helpful."
            ),
        },
    ]
    return llm.chat(messages, model=cfg["model"], **cfg.get("params", {}))


# ---------------------------
# Evidence normalizer
# ---------------------------

def _flatten_evidence_for_synth(evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Turn mixed tool outputs into RAG-ready docs that synthesize_answer can consume.

    Acceptable inputs (from ToolExecutor):
      - {"kind":"docs","value":[{text,source,url, title?},...]}
      - {"kind":"text","value":"..."}
      - {"kind":"json","value":[{...}, {...}]}
      - {"kind":"error","value":"..."}
    Output: List[{title, url, content, source}]
    """
    docs: List[Dict[str, str]] = []

    for item in evidence or []:
        kind = item.get("kind")
        src = item.get("source") or item.get("tool") or "tool"
        val = item.get("value")

        if kind == "docs" and isinstance(val, list):
            for d in val:
                docs.append({
                    "title": d.get("title") or d.get("source") or "doc",
                    "url": d.get("url") or "",
                    "content": d.get("text") or "",
                    "source": d.get("source") or src,
                })

        elif kind == "text" and isinstance(val, str):
            docs.append({
                "title": src,
                "url": "",
                "content": val[:DEFAULT_TEXT_SNIPPET_LEN],
                "source": src,
            })

        elif kind == "json":
            try:
                compact = json.dumps(val, ensure_ascii=False)[:DEFAULT_JSON_SNIPPET_LEN]
            except Exception:
                compact = str(val)[:DEFAULT_JSON_SNIPPET_LEN]
            docs.append({
                "title": f"{src} (json)",
                "url": "",
                "content": compact,
                "source": src,
            })

        elif kind == "error":
            docs.append({
                "title": f"{src} (error)",
                "url": "",
                "content": str(val),
                "source": src,
            })

        else:
            docs.append({
                "title": f"{src} ({kind or 'unknown'})",
                "url": "",
                "content": str(val)[:DEFAULT_TEXT_SNIPPET_LEN],
                "source": src,
            })

    return docs[:MAX_DOCS_FOR_SYNTH]


# ---------------------------
# Main entry: plan → execute → synthesize
# ---------------------------

def handle_query_planned(user_question: str, *, show_summary: bool = True):
    """
    Single entrypoint that replaces the old waterfall.

    Returns:
      answer: str
      steps: List[planned step dicts]
      trace: Dict of per-step execution info
      docs_for_synth: List[{title,url,content,source}]
    """
    # 1) Plan
    planner = ToolPlanner(get_llm)
    steps = planner.plan(user_question, tool_catalog)

    if show_summary:
        print("\n=== Planned Execution Summary ===")
        for s in steps:
            print(f"[{s['id']}] {s['tool']}")
            print(json.dumps(s.get("args", {}), indent=2))
            print("-" * 40)
        print("=================================\n")

    # 2) Execute
    executor = ToolExecutor(tool_catalog)
    trace, evidence = executor.run(steps)

    # 3) Normalize evidence for synth
    shaped = []
    for e in (evidence or []):
        shaped.append({**e, "source": e.get("source") or e.get("tool") or "tool"})
    docs_for_synth = _flatten_evidence_for_synth(shaped)

    # 4) Synthesize answer using ONLY the above docs
    answer = synthesize_answer(user_question, docs_for_synth)

    return answer, steps, trace, docs_for_synth
