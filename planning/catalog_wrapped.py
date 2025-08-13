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
    today_iso = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
    prompt = f"""
        You are a strict parameter generator for the Fathom Meetings API.

        Given the user's request, produce ONLY valid JSON containing relevant params.
        The JSON must have exactly one top-level key: "params".
        Interpret all dates relative to today's date: {today_iso}.

        The "params" object may contain only the following keys:
        - created_after (ISO 8601 datetime string, e.g., "2025-08-01T00:00:00Z")
        - created_before (ISO 8601 datetime string)
        - meeting_type (string, must be exactly "external" or "internal")
        - calendar_invitees (list of full email addresses)
        - calendar_invitees_domains (list of domain strings, e.g., ["blvd.co"])
        - include_summary (boolean: true or false)
        - include_transcript (boolean: true or false)

        Rules:
        1. DO NOT include any keys that are not listed above.
        2. If the user does not specify a time range, default to:
        {{"created_after": "{default_after}"}}
        3. If a parameter is not relevant, omit it.
        4. Your entire output must be valid JSON with this shape:
        {{
            "params": {{
            ...
            }}
        }}

        User request: {user_query}
        """

    llm, cfg = get_llm(provider="openai", model="gpt-4o-mini")
    raw = llm.chat([{"role": "user", "content": prompt}],
                   model=cfg["model"], **cfg.get("params", {"temperature": 0}))
    try:
        parsed = json.loads(str(raw))
    except Exception:
        parsed = {"params": {}}
    sanitized = sanitize_fathom_params(parsed)

    # Enforce default if missing
    if "created_after" not in sanitized["params"]:
        sanitized["params"]["created_after"] = default_after

    return sanitized

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
                # Start with any existing params from args
                merged = dict(args.get("params", {}))

                # Merge stray args into params only if they are in ALLOWED_PARAMS
                for key, val in args.items():
                    if key != "params" and key in ALLOWED_PARAMS:
                        merged[key] = val

                args["params"] = merged

                # If still no params, generate from LLM
                if not args["params"] and qa:
                    args["params"] = fathom_param_generator(qa.raw_query)["params"]
                print(qa, args)
                return _orig(args)

            run = run_with_llm

        elif k == "typesense_docs_live":
            run = wrap_tool(run, needs_ngrams)

        else:
            run = run

        wrapped[k] = {**v, "run_wrapped": run}

    return wrapped
