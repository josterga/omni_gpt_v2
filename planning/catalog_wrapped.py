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

ALLOWED_PARAMS = {
    "created_after",
    "created_before",
    "meeting_type",
    "calendar_invitees",
    "calendar_invitees_domains",
    "include_summary",
    "include_transcript",
}

def sanitize_fathom_params(llm_output: dict) -> dict:
    if "params" not in llm_output or not isinstance(llm_output["params"], dict):
        return {"params": {}}
    params = {k: v for k, v in llm_output["params"].items() if k in ALLOWED_PARAMS}
    return {"params": params}

def fathom_param_generator(user_query: str) -> dict:
    default_after = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
    prompt = f"""
You are a strict parameter generator for the Fathom Meetings API.

Given the user's request, produce ONLY valid JSON with exactly one top-level key: "params".

The "params" object may contain only:
- created_after (ISO datetime string)
- created_before (ISO datetime string)
- meeting_type ("external" or "internal")
- calendar_invitees (list of emails)
- calendar_invitees_domains (list of domains)
- include_summary (boolean)
- include_transcript (boolean)

If the request has no date range, default to:
  {{"created_after": "{default_after}"}}

If a parameter is not relevant, omit it.

User request: {user_query}
"""

    llm, cfg = get_llm(provider="openai", model="gpt-4o-mini")
    raw = llm.chat([{"role": "user", "content": prompt}],
                   model=cfg["model"], **cfg.get("params", {"temperature": 0}))
    try:
        parsed = json.loads(str(raw))
    except Exception:
        parsed = {"params": {}}
    return sanitize_fathom_params(parsed)

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
            params = fathom_param_generator(
                getattr(qa, "raw_query", "") or getattr(qa, "original_query", "") or str(qa)
            )["params"]

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
            def run_with_llm(args, qa=None, _orig=v["run"]):
                # If params are missing, try to generate them from raw query
                if not args.get("params") and qa:
                    args["params"] = fathom_param_generator(qa.raw_query)["params"]
                return _orig(args)

            run = run_with_llm

        elif k == "typesense_docs_live":
            run = wrap_tool(run, needs_ngrams)

        else:
            run = run

        wrapped[k] = {**v, "run_wrapped": run}

    return wrapped
