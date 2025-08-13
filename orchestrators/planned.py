import json
from planning.planner import ToolPlanner
from planning.executor import ToolExecutor
from planning.catalog_wrapped import build_wrapped_catalog
from tooling.query_artifacts import LazyQueryArtifacts
from applied_ai.keyword_extraction.keyword_extractor.extractor import KeywordExtractor
from applied_ai.keyword_extraction.keyword_extractor.stopword_pruner import prune_stopwords_from_results
from applied_ai.chunking.chunking.pipeline import run_chunking
from app_core import is_metric_query, ngram_config
from applied_ai.generation.generation.router import get_llm
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

    # Filter for planner
    filtered = {k: v for k, v in catalog.items() if k in allowed_tool_ids}

    # Bind qa into each tool (avoid late-binding bug)
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

    # Some executors only put structured outputs in trace
    for sid, t in (trace or {}).items():
        out = t.get("output")
        if isinstance(out, dict):
            # preserve tool name for source
            raw_items.append({**out, "tool": t.get("tool") or sid})

    # ---- Normalize into docs for synthesis ----
    normed_docs = []
    for ev in (raw_items or []):
        # Unwrap if the tool stuffed result under "output"
        payload = ev.get("output") if isinstance(ev.get("output"), dict) else ev
        if not isinstance(payload, dict):
            # string or something else → make it text
            normed_docs.append({
                "title": ev.get("tool") or "tool",
                "url": "",
                "content": str(payload),
                "source": ev.get("source") or ev.get("tool") or "tool",
            })
            continue

        source = payload.get("source") or ev.get("tool") or "tool"
        kind = payload.get("kind") or ev.get("kind") or "text"
        val = payload.get("value")

        if kind == "docs" and isinstance(val, list):
            for d in val:
                if not isinstance(d, dict):
                    continue
                normed_docs.append({
                    "title": d.get("title") or source or "doc",
                    "url": d.get("url") or "",
                    "content": d.get("text") or d.get("chunk_text") or "",
                    "source": d.get("source") or source,
                })

        elif kind == "text":
            text = val if isinstance(val, str) else str(val)
            if text.strip():
                normed_docs.append({
                    "title": source,
                    "url": "",
                    "content": text,
                    "source": source,
                })

        elif kind == "json":
            try:
                compact = json.dumps(val, ensure_ascii=False)
            except Exception:
                compact = str(val)
            normed_docs.append({
                "title": f"{source} (json)",
                "url": "",
                "content": compact[:800],
                "source": source,
            })

    # Debug aid
    print(f"[planned] normed_docs={len(normed_docs)} | nonempty_contents={sum(1 for d in normed_docs if (d.get('content') or '').strip())}")
    if normed_docs:
        first = normed_docs[0]
        print(f"[planned] sample: title={first.get('title')} url={first.get('url')} content_len={len(first.get('content',''))}")

    # ---- Synthesize ----
    answer = synthesize_answer(query, normed_docs, provider="openai", model="gpt-4o-mini")
    return answer, steps, trace, normed_docs