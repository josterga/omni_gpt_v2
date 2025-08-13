import json
from planning.planner import ToolPlanner
from planning.executor import ToolExecutor
from planning.catalog_wrapped import build_wrapped_catalog
from tooling.query_artifacts import LazyQueryArtifacts
from import_shims import KeywordExtractor, prune_stopwords_from_results, run_chunking, get_llm
from app_core import is_metric_query, ngram_config
from evidence import flatten_for_synth  # <-- import the unified flattener
from synthesis import synthesize_answer

def make_lazy_artifacts(query: str) -> LazyQueryArtifacts:
    def extractor():
        ke = KeywordExtractor(ngram_config)
        ngrams_obj = ke.extract(query)
        pruned = prune_stopwords_from_results(
            ngrams_obj,
            set(ngram_config["ngram"]["stopwords"]),
            "remove_stopwords_within_phrase"
        )
        return pruned.get("ngram") if isinstance(pruned, dict) else pruned

    def embedding_builder():
        chunks = run_chunking(
            raw_text=query,
            chunk_method="sentence",
            max_tokens=300,
            overlap_tokens=40,
            inject_headers=True,
            provider="voyage",
            model_name="voyage-3.5",
        )
        return {"chunks": chunks, "embedding": chunks[0]["embedding"] if chunks else None}

    return LazyQueryArtifacts(query, extractor, embedding_builder)


def run(query: str, *, allowed_tool_ids: list[str]):
    qa = make_lazy_artifacts(query)
    catalog = build_wrapped_catalog(is_metric_query, mode="planned")

    filtered = {k: v for k, v in catalog.items() if k in allowed_tool_ids}

    bound = {}
    for k, v in filtered.items():
        def make_run(vv):
            return lambda args, qa=qa: vv["run_wrapped"](args, qa=qa)
        bound[k] = {**v, "run": make_run(v)}

    planner = ToolPlanner(get_llm)
    steps = planner.plan(query, bound)
    executor = ToolExecutor(bound)
    trace, evidence = executor.run(steps)

    raw_items = []
    if evidence:
        raw_items.extend(evidence)

    for sid, t in (trace or {}).items():
        out = t.get("output")
        if isinstance(out, dict):
            raw_items.append({
                "kind": out.get("kind", "json"),
                "value": out.get("value"),
                "tool": t.get("tool") or sid,
                "source": out.get("source") or (t.get("tool") or sid)
            })



    print("\n[DEBUG] raw_items before flatten_for_synth")
    for i, item in enumerate(raw_items, 1):
        src = item.get("source") or item.get("tool")
        kind = item.get("kind")
        val_type = type(item.get("value")).__name__
        print(f"{i}. tool={src} kind={kind} value_type={val_type}")
        if src == "slack_search":
            print("    value:", item.get("value"))

    # ---- Flatten into docs for synthesis ----
    normed_docs = flatten_for_synth(raw_items, mode="planned")

    print(f"[planned] flattened_docs={len(normed_docs)} | nonempty_contents={sum(1 for d in normed_docs if (d.get('content') or '').strip())}")
    if normed_docs:
        first = normed_docs[0]
        print(f"[planned] sample: title={first.get('title')} url={first.get('url')} content_len={len(first.get('content',''))}")

    context_preview = "\n\n".join(
    f"{d.get('source') or ''} | {d.get('title') or ''}:\n{d.get('content') or ''}"
    for d in normed_docs
    )
    token_est = len(context_preview.split())  # crude token estimate
    print(f"[planned] context approx {token_est} words / {token_est//0.75:.0f} tokens")
    print(f"[planned] first 500 chars of context:\n{context_preview[:500]}")
    # ---- Synthesize ----
    answer = synthesize_answer(
        query,
        normed_docs,
        provider="openai",
        model="gpt-5",  # large-context model
        params={
            "max_completion_tokens": 6000
        }
    )
    return answer, steps, trace, normed_docs
