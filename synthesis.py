# synthesis.py
from typing import List, Dict, Any
from import_shims import get_llm

def synthesize_answer(query: str,
                      docs: List[Dict[str, Any]],
                      provider: str,
                      model: str,
                      params: dict = None) -> str:
    # docs are already flattened
    lines = []
    for d in docs or []:
        title = d.get("title") or d.get("source") or "doc"
        url = d.get("url") or ""
        content = d.get("content") or ""
        src = d.get("source") or ""
        if content.strip():
            lines.append(f"{src} | {title} ({url}):\n{content}")
    context = "\n\n".join(lines)

    # If context is empty, force an honest "insufficient context" reply.
    if not context.strip():
        return (
            "I don't have supporting context to answer yet. "
            "Please include relevant Slack messages, docs, Community, or MCP results."
        )

    try:
        llm, cfg = get_llm(
            provider=provider,
            model=model,
            params=params or {}  # <- forward planned mode params
        )
        
        # Check if LLM is available
        if llm is None:
            return _fallback_synthesis(query, context)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant for Omni Analytics. You are given context from multiple sources. "
                    "Use ONLY the provided context. If the context is insufficient, say exactly what is missing."
                )
            },
            {
                "role": "user",
                "content": (
                    "Important Instructions:\n"
                    "- Use only the provided context. Do not hallucinate.\n"
                    "- Cite where facts come from (source) using the provided source labels.\n"
                    "- Follow the structure: Answer, Source Highlights, Unanswered Questions.\n"
                    "\n---\n\n"
                    f"User Question:\n{query}\n\n"
                    "---\n\n"
                    f"Available Information:\n{context}\n\n"  # <<<<<<<<<<<<<< use context, not {docs}
                    "---\n\n"
                    "Answer Format:\n"
                    "1. **Answer**  \n"
                    "- Summarize the correct response in a clear, human-readable way.  \n"
                    "- Use only the information from the provided context.  \n"
                    "- If there are multiple answers to the question, include all answers.  \n"
                    "- Do **not** hallucinate or add unstated assumptions.\n"
                    "- If the question or context involves structured data (e.g., YAML, JSON, config files, code), include an **example derived from the context** formatted in a fenced code block. Do not hallucinate YAML or code.\n\n"
                    "2. **Source Highlights**  \n"
                    "- List key facts or data points from the sources that directly support the answer.  \n"
                    "- Do not restate entire paragraphs.\n\n"
                    "3. **Unanswered Questions** *(if applicable)*  \n"
                    "- Note any aspects of the user's question that the provided information does **not** answer.  \n"
                    "- Be concise but honest about the gap.\n"
                    "- If there are no unanswered questions, don't include this section as part of your answer.\n\n"
                    "Now write your answer."
                )
            }
        ]
        return llm.chat(messages, model=cfg["model"], **cfg.get("params", {}))
        
    except Exception as e:
        print(f"⚠️  LLM synthesis failed: {e}, using fallback synthesis")
        return _fallback_synthesis(query, context)

def _fallback_synthesis(query: str, context: str) -> str:
    """Fallback synthesis when LLM is not available."""
    return f"""**Answer**
Based on the available information, I can provide a summary of what I found:

{context[:500]}{'...' if len(context) > 500 else ''}

**Source Highlights**
- Information was found from the provided sources
- Context length: {len(context)} characters

**Unanswered Questions**
I was unable to provide a complete AI-generated synthesis because the language model is not currently available. The raw information above may still be helpful for answering your question: "{query}"

Please try again later when the AI service is available, or review the source information directly."""
