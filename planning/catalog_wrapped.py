from planning.catalog import tool_catalog as BASE_CATALOG
from tooling.decorators import (
    needs_ngrams,
    needs_embedding,
    with_slack_exclusions,
    gated_by_is_metric,
)
from applied_ai.generation.generation.router import get_llm
import json
from datetime import datetime, timedelta

def wrap_tool(fn, *wrappers):
    """
    Apply wrappers in the order given.
    Outer wrapper is applied first.
    """
    for wrapper in wrappers:
        fn = wrapper(fn)
    return fn

def generate_fathom_params_from_query(query: str) -> dict:
    """
    Use an LLM to produce a params dict for fathom_list_meetings.
    """
    llm, cfg = get_llm(provider="openai", model="gpt-4o-mini")
    prompt = f"""
You are a parameter generator for the Fathom Meetings API.
Given the user's request, output ONLY a JSON object for the 'params' argument to list_meetings().

Available params:
- created_after (ISO datetime)
- created_before (ISO datetime)
- meeting_type ("external" or "internal")
- calendar_invitees (list of emails)
- calendar_invitees_domains (list of email domains)
- include_summary (bool)
- include_transcript (bool)

Only include the parameters relevant to the request. 
User request: {query}
"""
    raw = llm.chat([{"role": "user", "content": prompt}],
                   model=cfg["model"], **cfg.get("params", {"temperature": 0}))
    try:
        parsed = json.loads(str(raw).strip())
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}

def wrap_fathom_tool(fn, mode):
    """
    Wrap fathom_list_meetings so that:
    - Direct mode: skipped entirely (no API call)
    - Planned mode: params are generated via LLM if not provided
    """
    def wrapped(args, qa=None):
        if mode != "planned":
            return {
                "kind": "text",
                "value": "[Fathom skipped — only available in planned mode]",
                "preview": "fathom:skipped"
            }

        params = args.get("params", {})
        if not params and qa:
            params = generate_fathom_params_from_query(
                getattr(qa, "query", None) or getattr(qa, "original_query", "") or str(qa)
            )

        # Default to last 14 days if no date range
        if "created_after" not in params:
            params["created_after"] = (datetime.utcnow() - timedelta(days=14)).isoformat() + "Z"

        args["params"] = params
        return fn(args)
    return wrapped

def build_wrapped_catalog(is_metric_fn, mode="direct"):
    """
    Returns a catalog where each tool has a `run_wrapped(args, qa=LazyQueryArtifacts)`
    that will inject only the required query artifacts (ngrams, embedding, etc.)
    before running the original tool function.
    """
    wrapped = {}

    for k, v in BASE_CATALOG.items():
        run = v["run"]

        if k == "slack_search":
            run = wrap_tool(run, with_slack_exclusions, needs_ngrams)

        elif k == "typesense_search":
            run = wrap_tool(run, needs_ngrams)

        elif k == "docs_embed_search":
            run = wrap_tool(run, needs_embedding)

        elif k == "community_embed_search":
            run = wrap_tool(run, needs_embedding)

        elif k == "mcp_query":
            run = run  # gated by metric queries in its own wrapper elsewhere

        elif k == "fathom_list_meetings":
            def fathom_wrapped(args, qa=None, _run=run):
                # If query is not given, try generating params from raw query via LLM
                if "query" not in args and qa is not None:
                    query_text = getattr(qa, "raw_query", "")
                    args = {**args, **generate_fathom_params_from_query(query_text)}
                return _run(args)
            run = fathom_wrapped

        elif k == "typesense_docs_live":
            run = wrap_tool(run, needs_ngrams)

        else:
            run = run

        wrapped[k] = {**v, "run_wrapped": run}

    return wrapped
