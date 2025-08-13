from planning.catalog import tool_catalog as BASE_CATALOG
from tooling.decorators import (
    needs_ngrams,
    needs_embedding,
    with_slack_exclusions,
    gated_by_is_metric,
)
# If you later add Typesense live fetch or URL formatting, you can import:
# from tooling.decorators import with_live_fetch, with_url_canonicalization

def wrap_tool(fn, *wrappers):
    """
    Apply wrappers in the order given.
    Outer wrapper is applied first.
    """
    for wrapper in wrappers:
        fn = wrapper(fn)
    return fn

def build_wrapped_catalog(is_metric_fn):
    """
    Returns a catalog where each tool has a `run_wrapped(args, qa=LazyQueryArtifacts)`
    that will inject only the required query artifacts (ngrams, embedding, etc.)
    before running the original tool function.
    """
    wrapped = {}

    for k, v in BASE_CATALOG.items():
        run = v["run"]

        if k == "slack_search":
            # Slack search uses keyword-based matching + exclusions, no embeddings
            run = wrap_tool(run, with_slack_exclusions, needs_ngrams)

        elif k == "typesense_search":
            # Typesense uses keywords; you could also add with_live_fetch here
            run = wrap_tool(run, needs_ngrams)

        elif k == "docs_embed_search":
            # Docs use embedding search; no keyword extraction needed
            run = wrap_tool(run, needs_embedding)

        elif k == "community_embed_search":
            # Community (Discourse) search uses embeddings
            run = wrap_tool(run, needs_embedding)

        elif k == "mcp_query":
            # MCP gated by metric queries; no ngrams/embedding needed
            run = run
        elif k == "fathom_list_meetings":
            # No pre-processing needed
            run = run

        elif k == "typesense_docs_live":
            # This variant: keyword search + optional live fetch; can add URL cleanup
            run = wrap_tool(run, needs_ngrams)

        else:
            # Default: leave unchanged
            run = run

        wrapped[k] = {**v, "run_wrapped": run}

    return wrapped
