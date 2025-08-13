from planning.catalog_wrapped import build_wrapped_catalog
from tooling.query_artifacts import LazyQueryArtifacts
from evidence import flatten_for_synth
from synthesis import synthesize_answer
from import_shims import KeywordExtractor, prune_stopwords_from_results, run_chunking
from app_core import is_metric_query

ngram_config = {
    "strategy": "ngram",
    "ngram": {
        "ngram_sizes": [1, 2, 3],
        "stopwords": [
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
            "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
            "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
            "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
            "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
            "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
            "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
            "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your",
            "yours", "yourself", "yourselves"
        ]
    }
}

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
    catalog = build_wrapped_catalog(is_metric_query, mode="direct")

    all_evidence, trace = [], {}
    for tool_id in allowed_tool_ids:
        tool = catalog.get(tool_id)
        if not tool:
            trace[tool_id] = {"ok": False, "error": "not in catalog"}
            continue
        try:
            ev = tool["run_wrapped"]({"query": query}, qa=qa)
            ev["tool"] = tool_id
            ev["source"] = ev.get("source") or tool_id
            all_evidence.append(ev)
            trace[tool_id] = {"ok": True, "kind": ev.get("kind"), "preview": ev.get("preview")}
        except Exception as e:
            all_evidence.append({"kind": "error", "value": str(e), "source": tool_id})
            trace[tool_id] = {"ok": False, "error": str(e)}

    docs_for_synth = flatten_for_synth(all_evidence, mode="direct")
    answer = synthesize_answer(query, docs_for_synth, provider="openai", model="gpt-4o-mini")
    steps = [{"id": f"run:{tid}", "tool": tid, "args": {"query": query}} for tid in allowed_tool_ids if tid in catalog]
    return answer, steps, trace, docs_for_synth
