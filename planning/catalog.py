from typing import Dict
from .types import ToolCatalog
from .helpers import load_json_embeddings, search_json_chunks, search_typesense_ngrams, fetch_live_content

# Import your existing building blocks
try:
    # Dev: monorepo namespace
    from applied_ai.generation.generation.router import get_llm
    from applied_ai.keyword_extraction.keyword_extractor.extractor import KeywordExtractor
    from applied_ai.keyword_extraction.keyword_extractor.stopword_pruner import prune_stopwords_from_results
    from applied_ai.slack_search.slack_search.searcher import SlackSearcher
    from applied_ai.mcp_client.mcp_client.registry import MCPRegistry
    from applied_ai.retrieval.retrieval.faiss_retriever import FAISSRetriever
    from applied_ai.chunking.chunking.pipeline import run_chunking
    from fathom_module import fathom_api
except ImportError:
    # Prod: packages installed from Git subdirectories (no namespace)
    from generation.router import get_llm
    from keyword_extractor.extractor import KeywordExtractor                # ← correct
    from keyword_extractor.stopword_pruner import prune_stopwords_from_results
    from slack_search.searcher import SlackSearcher
    from retrieval.faiss_retriever import FAISSRetriever
    from chunking.pipeline import run_chunking
    from mcp_client.registry import MCPRegistry  
# Example environment wiring (optional)

import os
SLACK_TOKEN = os.getenv("SLACK_API_KEY")
OMNI_MODEL_ID = os.getenv("OMNI_MODEL_ID")
OMNI_API_KEY = os.getenv("OMNI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
ENABLE_MCP = os.getenv("ENABLE_MCP", "false").lower() in {"1", "true", "yes"}
docs_chunks = "sources/docs/docs-000.jsonl"
community_chunks = "sources/discourse/discourse-000.jsonl"
mcp_client = None
if ENABLE_MCP:
    registry = MCPRegistry()
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # or get_llm if needed
    mcp_client = registry.get_client(
        "Omni",
        url=BASE_URL,
        headers={
            "Authorization": f"Bearer {OMNI_API_KEY}",
            "Accept": "application/json, text/event-stream",
            "X-MCP-MODEL-ID": OMNI_MODEL_ID
        },
        openai_client=openai_client   # ✅ pass it in
    )
else:
    class _NoMCP:
        def run_agentic_inference(self, *a, **k): 
            return {"answer": "[MCP disabled]", "reasoning_steps": []}
    mcp_client = _NoMCP()

tool_catalog: ToolCatalog = {
    "slack_search": {
        "name": "slack_search",
        "description": "Search Slack messages/threads and return relevant snippets.",
        "produces": "docs",
        "run": lambda args: {
            "kind": "docs",
            "value": [
                {
                    "text": r.get("text", "") if isinstance(r, dict) else str(r),
                    "source": "slack",
                    "url": (r.get("metadata", {}) or {}).get("permalink")
                }
                for r in SlackSearcher(
                    slack_token=SLACK_TOKEN,
                    result_limit=args.get("limit", 5),
                    thread_limit=args.get("thread_limit", 5)
                ).search(args["query"])
            ],
            "preview": f"slack:{args.get('query','')[:60]}"
        }
    },
    # "typesense_search": {
    #     "name": "typesense_search",
    #     "description": "Search Omni docs via Typesense and fetch live content.",
    #     "produces": "docs",
    #     "run": lambda args: {
    #         "kind": "docs",
    #         "value": [
    #             {"text": d["content"], "source": "docs", "url": d["url"]}
    #             for d in search_typesense_ngrams(
    #                 {"ngram": [args["query"]]}, max_results=args.get("max_results", 5)
    #             )
    #         ],
    #         "preview": f"typesense:{args.get('query','')[:60]}"
    #     }
    # },
    "docs_embed_search": {
        "name": "docs_embed_search",
        "description": "Semantic search over embedded JSON chunks.",
        "produces": "docs",
        "run": lambda args: (lambda chunks, qembed: {
            "kind": "docs",
            "value": [
                {
                    "text": c.get("chunk_text",""), "source": "docs",
                    "url": f"https://docs.omni.co/{c['metadata'].get('path','').replace('./docs/','',1).removesuffix('.md')}"
                }
                for c in search_json_chunks(qembed, chunks, top_k=args.get("top_k", 5))
            ],
            "preview": "docs_embed:ok"
        })(
            load_json_embeddings(docs_chunks),
            run_chunking(
                raw_text=args["query"], chunk_method="sentence",
                max_tokens=300, overlap_tokens=40,
                inject_headers=True, provider="voyage", model_name="voyage-3.5"
            )[0]["embedding"]
        )
    },
    "mcp_query": {
        "name": "mcp_query",
        "description": "Ask MCP for a synthesized answer w/ reasoning.",
        "produces": "text",
        "run": lambda args: (lambda res: {
            "kind": "text",
            "value": res.get("answer",""),
            "preview": res.get("answer","")[:280],
            "metadata": {"reasoning": res.get("reasoning_steps", [])}
        })(mcp_client.run_agentic_inference(
        args.get("query") or args.get("question") or ""
    ))
    },
    "fathom_list_meetings": {
    "name": "fathom_list_meetings",
    "description": (
        "List meeting records from the Fathom API. "
        "Always limit the number of meetings returned to the smallest set needed to answer the question. "
        "Prefer recent time windows (e.g., last 7–14 days) unless the question specifies a longer period. "
        "Args: 'params' dict can include filters such as:\n"
        "  - created_after (string, ISO 8601): Only include meetings after this date, "
        "e.g. '2024-07-01T00:00:00Z'.\n"
        "  - created_before (string, ISO 8601): Only include meetings before this date.\n"
        "  - meeting_type (string): Filter by type, values are ONLY 'external' or 'internal'.\n"
        "  - participants (string or list): Filter meetings by participant name(s).\n"
        "  - calendar_invitees_domains (list): Restrict to invitees from certain domains, "
        "e.g. ['sumup.com'].\n"
        "  - include_summary (boolean): Whether to include meeting summaries.\n"
        "  - include_transcript (boolean): Whether to include full transcripts. \n"
    ),
    "produces": "json",
    "run": lambda args: (lambda meetings: {
        "kind": "json",
        "value": meetings,
        "preview": f"fathom meetings: {len(meetings)} found"
    })(
        list(fathom_api.list_meetings(params=args.get("params", {})))
    )
}
}