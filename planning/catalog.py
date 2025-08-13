import os
from pathlib import Path
from typing import List, Dict, Any

# Monorepo or prod import shim
try:
    from applied_ai.slack_search.slack_search.searcher import SlackSearcher
    from applied_ai.chunking.chunking.pipeline import run_chunking
    from applied_ai.mcp_client.mcp_client.registry import MCPRegistry

except ImportError:
    from slack_search.searcher import SlackSearcher
    from chunking.pipeline import run_chunking
    from mcp_client.registry import MCPRegistry


from app_core import load_json_embeddings, search_json_chunks
from tooling.common_utils import make_docs_url_from_path, make_community_url
from fathom_module import fathom_api
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# === MCP client init ===
OMNI_MODEL_ID = os.getenv("OMNI_MODEL_ID")
OMNI_API_KEY = os.getenv("OMNI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENABLE_MCP = os.getenv("ENABLE_MCP", "true").lower() in {"1", "true", "yes"}

if ENABLE_MCP:
    registry = MCPRegistry()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    mcp_client = registry.get_client(
        "Omni",
        openai_client=openai_client,
        url=BASE_URL,
        headers={
            "Authorization": f"Bearer {OMNI_API_KEY}",
            "Accept": "application/json, text/event-stream",
            "X-MCP-MODEL-ID": OMNI_MODEL_ID
        }
    )
else:
    class _NoMCP:
        def run_agentic_inference(self, *a, **k):
            return {"answer": "[MCP disabled]", "reasoning_steps": []}
    mcp_client = _NoMCP()

# === File paths for embeddings ===
DOCS_EMBED_FILE = Path("sources/docs/docs-000.jsonl")
COMMUNITY_EMBED_FILE = Path("sources/discourse/discourse-000.jsonl")

# === Helper for Docs & Community ===
def _search_embeddings_for_source(
    query_embedding: List[float],
    chunks_file: Path,
    source: str,
    url_formatter=None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    chunks = load_json_embeddings(chunks_file)
    top_chunks = search_json_chunks(query_embedding, chunks, top_k=top_k)

    results = []
    seen_urls = set()
    for c in top_chunks:
        if source == "docs":
            url = url_formatter(c["metadata"].get("path", ""))
            title = url
        elif source == "community":
            slug = str(c["metadata"].get("slug") or "")
            topic_id = str(c["metadata"].get("topic_id") or "")
            url = url_formatter(slug, topic_id) if url_formatter else None
            title = slug or "Community"
        else:
            url = None
            title = c.get("title") or source

        if url in seen_urls:
            continue
        seen_urls.add(url)

        results.append({
            "text": c.get("chunk_text", ""),
            "source": source,
            "url": url,
            "title": title,
        })

    return results

# === Slack token ===
SLACK_TOKEN = os.getenv("SLACK_API_KEY")

# === Main tool catalog (unwrapped) ===
tool_catalog = {
    # Slack search (works with needs_ngrams + with_slack_exclusions)
    "slack_search": {
        "name": "slack_search",
        "description": "Search Slack messages/threads and return relevant snippets. Great for recent decisions, tribal knowledge, and ephemeral fixes. You can include filters for in:#<channel> in the query. Every customer has their own channel using omni-<customer> as the name. Internal product feature-related channels: ai, ai-development, api, backend, calcs, dashboards, dbt, disco-stew, docs, drafts-branches, embed, eng, event-loop-lag, it-help, modeling, omni-omni-analytics, product, product-promise-requests, proj-ai-docs, proj-csv-upload, scheduled-deliveries-and-alerts,  spreadsheets, ux, visualizations. Sales/Marketing-related channels: sales, marketing, closed-lost-notifications, sigma-compete, optys-qual-hall.",
        "produces": "docs",
        "run": lambda args, qa=None: {
            "kind": "docs",
            "value": [
                {
                    "text": r.get("text", ""),
                    "source": "slack",
                    "url": (r.get("metadata", {}) or {}).get("permalink")
                }
                for ng in args["ngrams"]
                for r in SlackSearcher(
                    slack_token=SLACK_TOKEN,
                    result_limit=args.get("limit", 5),
                    thread_limit=args.get("thread_limit", 5)
                ).search(ng)
            ],
            "preview": f"slack:{args.get('query','')[:60]}"
        }
    },

    # Docs embedding search
    "docs_embed_search": {
        "name": "docs_embed_search",
        "description": "Semantic search over embedded Omni Docs JSON chunks. Best for official processes, architecture, deployment guides.",
        "produces": "docs",
        "run": lambda args, qa=None: {
            "kind": "docs",
            "value": _search_embeddings_for_source(
                query_embedding=args["query_embedding"],
                chunks_file=DOCS_EMBED_FILE,
                source="docs",
                url_formatter=make_docs_url_from_path,
                top_k=args.get("top_k", 5),
            ),
            "preview": "docs_embed:ok",
        },
    },

    # Community embedding search
    "community_embed_search": {
        "name": "community_embed_search",
        "description": "Semantic search over embedded Community (Discourse) JSON chunks. Contains articles covering best practices, how-tos, and common patterns.",
        "produces": "docs",
        "run": lambda args, qa=None: {
            "kind": "docs",
            "value": _search_embeddings_for_source(
                query_embedding=args["query_embedding"],
                chunks_file=COMMUNITY_EMBED_FILE,
                source="community",
                url_formatter=make_community_url,
                top_k=args.get("top_k", 5),
            ),
            "preview": "community_embed:ok",
        },
    },

    # MCP query
    "mcp_query": {
        "name": "mcp_query",
        "description": "Ask MCP for a synthesized answer w/ reasoning. MCP queries are passed to a LLM which generates a SQL query against the database for a particular topic. Topics are selected by the MCP server. Data may include: Github, Salesforce, Product Usage Data (Organizations, Users, Queries, Models, Features), Customer Support Data (Pylon).",
        "produces": "text",
        "run": lambda args, qa=None: (
        lambda query_text: (
            {"kind": "error", "value": "No query provided to MCP"}
            if not query_text.strip()
            else (lambda res: {
                "kind": "text",
                "value": res.get("answer", ""),
                "preview": res.get("answer", "")[:280],
                "metadata": {"reasoning": res.get("reasoning_steps", [])}
            })(mcp_client.run_agentic_inference(query_text))
        )
    )(
        # 1) Preferred: query or question key
        args.get("query")
        or args.get("question")
        # 2) Fallback: join all non-empty arg values
        or " ".join(
            str(v) for v in args.values()
            if v not in (None, "", {}, [], Ellipsis)
        )
    )
},

    # Fathom meetings
    "fathom_list_meetings": {
        "name": "fathom_list_meetings",
        "description": (
            "List meeting records from the Fathom API. "
            "Limit results to smallest set needed to answer the question."
        ),
        "produces": "json",
        "run": lambda args, qa=None: (lambda meetings: {
            "kind": "json",
            "value": meetings,
            "preview": f"fathom meetings: {len(meetings)} found"
        })(list(fathom_api.list_meetings(params=args.get("params", {}))))
    }
}
