from tooling.query_artifacts import LazyQueryArtifacts

def needs_ngrams(fn):
    def wrapped(args, qa: LazyQueryArtifacts, **kw):
        if "ngrams" not in args or not args["ngrams"]:
            args = {**args, "ngrams": qa.ngrams}
        return fn(args, qa=qa, **kw)
    return wrapped

def needs_embedding(fn):
    def wrapped(args, qa: LazyQueryArtifacts, **kw):
        if "query_embedding" not in args or args["query_embedding"] is None:
            args = {
                **args,
                "query_embedding": qa.query_embedding,
                "query_chunks": qa.query_chunks,
            }
        return fn(args, qa=qa, **kw)
    return wrapped

def with_slack_exclusions(fn):
    SLACK_EXCLUSIONS = (
        " -in:customer-sla-breach -in:customer-triage -in:support-overflow "
        "-in:omnis -in:customer-membership-alerts -in:vector-alerts "
        "-in:notifications-alerts -cypress -github -sentry -squadcast -syften "
        "-in:leadership -in:leaders"
    )
    def wrapped(args, qa: LazyQueryArtifacts, **kw):
        q = args.get("query") or qa.raw_query
        args = {**args, "query": q + SLACK_EXCLUSIONS}
        return fn(args, qa=qa, **kw)
    return wrapped

def gated_by_is_metric(is_metric_fn):
    def deco(fn):
        def wrapped(args, qa: LazyQueryArtifacts, **kw):
            q = args.get("query") or qa.raw_query
            if not is_metric_fn(q):
                return {"kind": "text", "value": "", "preview": "mcp:skipped(non-metric)"}
            return fn(args, qa=qa, **kw)
        return wrapped
    return deco
