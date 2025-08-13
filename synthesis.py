# synthesis.py
from typing import List, Dict, Any
from applied_ai.generation.generation.router import get_llm

def synthesize_answer(query: str,
                      docs: List[Dict[str, Any]],
                      provider: str,
                      model: str) -> str:
    # docs are already flattened: {title,url,content,source}
    # Build readable context from *content* — not the raw docs object.
    lines = []
    for d in docs or []:
        title = d.get("title") or d.get("source") or "doc"
        url = d.get("url") or ""
        content = d.get("content") or ""
        src = d.get("source") or ""
        if content.strip():
            lines.append(f"{src} | {title} ({url}):\n{content}")
    context = "\n\n".join(lines)

    llm, cfg = get_llm(provider=provider, model=model)

    # If context is empty, force an honest “insufficient context” reply.
    if not context.strip():
        return (
            "I don’t have supporting context to answer yet. "
            "Please include relevant Slack messages, docs, Community, or MCP results."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for Omni Analytics. "
                "Use ONLY the provided context. If the context is insufficient, say exactly what is missing."
            )
        },
        {
            "role": "user",
            "content": (
                "Important Instructions:\n"
                "- Use only the provided context. Do not hallucinate.\n"
                "- Cite where facts come from (Docs / Community / Slack / MCP) using the provided source labels.\n"
                "- Follow the structure: Answer, Source Highlights, Unanswered Questions.\n"
                "\n---\n\n"
                f"User Question:\n{query}\n\n"
                "---\n\n"
                f"Available Information:\n{context}\n\n"  # <<<<<<<<<<<<<< use context, not {docs}
                "---\n\n"
                "Answer Format:\n"
                "1) Answer (concise)\n"
                "2) Source Highlights (bullet key facts)\n"
                "3) Unanswered Questions (only if gaps remain)\n"
            )
        }
    ]
    return llm.chat(messages, model=cfg["model"], **cfg.get("params", {}))
