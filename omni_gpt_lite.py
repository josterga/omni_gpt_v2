import os
import json
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from applied_ai.generation.generation.router import get_llm
from applied_ai.keyword_extraction.keyword_extractor.extractor import KeywordExtractor
from applied_ai.keyword_extraction.keyword_extractor.stopword_pruner import prune_stopwords_from_results
import time
from applied_ai.slack_search.slack_search.searcher import SlackSearcher
from applied_ai.mcp_client.mcp_client.client import MCPClient 
from applied_ai.mcp_client.mcp_client.registry import MCPRegistry
import openai



# If you want to use your own keyword-extraction later, you can import it here

load_dotenv()

TYPESENSE_API_KEY = os.getenv('TYPESENSE_API_KEY')
TYPESENSE_BASE_URL = os.getenv('TYPESENSE_BASE_URL')
TYPESENSE_URL = f"https://{TYPESENSE_BASE_URL}-1.a1.typesense.net/multi_search?x-typesense-api-key={TYPESENSE_API_KEY}"
TYPESENSE_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "text/plain"
}
TYPESENSE_SEARCH_BODY_TEMPLATE = {
    "searches": [
        {
            "collection": "omni-docs",
            "q": "",  # <-- user query will be filled in
            "query_by": "hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3,hierarchy.lvl4,hierarchy.lvl5,hierarchy.lvl6,content",
            "include_fields": "hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3,hierarchy.lvl4,hierarchy.lvl5,hierarchy.lvl6,content,anchor,url,type,id",
            "highlight_full_fields": "hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3,hierarchy.lvl4,hierarchy.lvl5,hierarchy.lvl6,content",
            "group_by": "url",
            "group_limit": 3,
            "sort_by": "item_priority:desc",
            "snippet_threshold": 8,
            "highlight_affix_num_tokens": 4,
            "filter_by": ""
        }
    ]
}
OMNI_MODEL_ID = os.getenv("OMNI_MODEL_ID")
OMNI_API_KEY = os.getenv("OMNI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENABLE_MCP = os.getenv("ENABLE_MCP", "true").lower() in {"1", "true", "yes"}


openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
if ENABLE_MCP:
    registry = MCPRegistry()
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
    mcp_client = None

def search_typesense(query, max_results=5):
    body = TYPESENSE_SEARCH_BODY_TEMPLATE.copy()
    body["searches"][0] = body["searches"][0].copy()
    body["searches"][0]["q"] = query

    resp = requests.post(
        TYPESENSE_URL,
        headers=TYPESENSE_HEADERS,
        data=json.dumps(body),
        timeout=8
    )
    resp.raise_for_status()
    resp_json = resp.json()
    results = resp_json.get("results", [])
    if not results:
        return []

    grouped_hits = results[0].get("grouped_hits", [])
    docs = []
    for group in grouped_hits:
        hits = group.get("hits", [])
        for hit in hits:
            doc = hit["document"]
            # Optionally parse HTML content fields
            content = doc.get("content", "")
            if content:
                soup = BeautifulSoup(content, "html.parser")
                content = soup.get_text(" ", strip=True)
            url = doc.get("url", "")
            # Build a readable title from hierarchy fields
            title_parts = [doc.get(f"hierarchy.lvl{i}") for i in range(7) if doc.get(f"hierarchy.lvl{i}")]
            title = " > ".join(title_parts)
            docs.append({
                "title": title,
                "url": url,
                "content": content
            })
            if len(docs) >= max_results:
                return docs
    return docs

def synthesize_answer(user_query, docs, llm_provider="openai", llm_model="gpt-4o-mini"):
    # Compose context from docs
    context = "\n\n".join([f"{doc['title']} ({doc['url']}):\n{doc['content']}" for doc in docs])
    llm, cfg = get_llm(provider=llm_provider, model=llm_model)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for Omni Analytics. You are answering the user's question using ONLY the provided information, "
                "which includes Slack conversations, official documentation, and community (discourse) discussions."
            )
        },
        {
            "role": "user",
            "content": (
                "Important Instructions:\n"
                "- Use only the provided context. **Do not hallucinate** or invent facts.\n"
                "- Clearly cite where each point comes from (e.g., 'Slack', 'Documentation', 'Community').\n"
                "- Follow the answer structure below.\n"
                "\n---\n\n"
                f"**User Question**:\n{user_query}\n\n"
                "---\n\n"
                f"**Available Information**:\n{context}\n\n"
                "---\n\n"
                "**Answer Format**:\n\n"
                "1. **Answer**  \n"
                "- Summarize the correct response in a clear, human-readable way.  \n"
                "- Use only the information from the provided context.  \n"
                "- Do **not** hallucinate or add unstated assumptions.\n"
                "- If the question or context involves structured data (e.g., YAML, JSON, config files, code), include a **generic example** formatted in a fenced code block.\n\n"
                "2. **Source Highlights**  \n"
                "- List key facts or data points from the sources that directly support the answer.  \n"
                "- Do not restate entire paragraphs.\n\n"
                "3. **Unanswered Questions** *(if applicable)*  \n"
                "- Note any aspects of the user's question that the provided information does **not** answer.  \n"
                "- Be concise but honest about the gap.\n"
                "- If there are no unanswered questions, don't include this section as part of your answer.\n\n"
                "Now write your answer."
            )
        }
    ]
    response = llm.chat(messages, model=cfg["model"], **cfg.get("params", {}))
    return response

def fetch_live_content(url, timeout=8):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # You may want to improve this to extract only the main content
        return soup.get_text(" ", strip=True)
    except Exception as e:
        print(f"[WARN] Failed to fetch live content from {url}: {e}")
        return None

def search_typesense_ngrams(ngrams, max_results=5):
    docs = []
    seen_urls = set()
    # Handle both dict and list input
    if isinstance(ngrams, dict) and "ngram" in ngrams:
        ngram_list = ngrams["ngram"]
    else:
        ngram_list = ngrams
    for ngram in ngram_list:
        # Prepare the search body for this ngram
        body = TYPESENSE_SEARCH_BODY_TEMPLATE.copy()
        body["searches"][0] = body["searches"][0].copy()
        body["searches"][0]["q"] = ngram

        resp = requests.post(
            TYPESENSE_URL,
            headers=TYPESENSE_HEADERS,
            data=json.dumps(body),
            timeout=8
        )
        resp.raise_for_status()
        resp_json = resp.json()
        results = resp_json.get("results", [])
        if not results:
            continue

        grouped_hits = results[0].get("grouped_hits", [])
        for group in grouped_hits:
            hits = group.get("hits", [])
            for hit in hits:
                doc = hit["document"]
                url = doc.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                # --- Fetch live content instead of using indexed content ---
                live_content = fetch_live_content(url)
                if not live_content:
                    # Fallback to indexed content if live fetch fails
                    content = doc.get("content", "")
                    if content:
                        soup = BeautifulSoup(content, "html.parser")
                        content = soup.get_text(" ", strip=True)
                else:
                    content = live_content
                title_parts = [doc.get(f"hierarchy.lvl{i}") for i in range(7) if doc.get(f"hierarchy.lvl{i}")]
                title = " > ".join(title_parts)
                docs.append({
                    "title": title,
                    "url": url,
                    "content": content
                })
                if len(docs) >= max_results:
                    return docs
                # Optional: be polite to servers
                time.sleep(0.5)
    return docs


def is_metric_query(query):
    metric_keywords = [
        "how many", "count", "number of", "average", "sum", "total", "list all", "show me", "statistics", "metric", "report",
        "github", "pylon", "omni", "Opportunities", "Accounts", "Build", "Global Cost", "Contract Search", "General Ledger", "Invoices", "Github Issues", "Pull Requests", "Workflow Runs", "Emoji Events", "Feature Flag", "Folders", "AI Usage", "Connections", "Document Views", "Feature Releases", "Models", "NPS", "Query History", "Query Presentations", "Users", "Harvest PS Time", "Harvest Users", "ARR Facts", "Contacts", "Customer Health", "Leads", "Opportunity History", "Tasks", "Pylon Issues", "Webhook Log Australia", "Webhook Log EastUSA", "Webhook Log Ireland"            # Add more keywords as needed
    ]
    q = query.lower()
    return any(kw in q for kw in metric_keywords)

if __name__ == "__main__":
    print("Type 'exit' or 'quit' to end the session.\n")
    slack_searcher = SlackSearcher(result_limit=5, thread_limit=5)

    while True:
        user_query = input("You: ")
        if user_query.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # --- Extract n-grams from user query ---
        ngram_config = {
            "strategy": "ngram",
            "ngram": {
                "ngram_sizes": [1, 2, 3],
                "stopwords": [
                    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
                    "could", "did", "do", "does", "doing", "down", "during",
                    "each", "few", "for", "from", "further",
                    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
                    "i", "if", "in", "into", "is", "it", "its", "itself",
                    "just",
                    "me", "more", "most", "my", "myself",
                    "no", "nor", "not", "now",
                    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
                    "same", "she", "should", "so", "some", "such",
                    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
                    "under", "until", "up",
                    "very",
                    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
                    "you", "your", "yours", "yourself", "yourselves"
                ]
            }
        }
        extractor = KeywordExtractor(ngram_config)
        ngrams = extractor.extract(user_query)
        ngrams = prune_stopwords_from_results(
            ngrams, set(ngram_config["ngram"]["stopwords"]), mode="remove_stopwords_within_phrase"
        )

        # --- Typesense search (all ngrams) ---
        typesense_docs = search_typesense_ngrams(ngrams)
        for doc in typesense_docs:
            doc["source"] = "typesense"

        # --- Slack search (2+ word ngrams) ---
        slack_ngrams = [ng for ng in ngrams if len(ng.split()) >= 2]
        slack_docs = []
        seen = set()
        for ngram in slack_ngrams:
            slack_results = slack_searcher.search(ngram)
            for res in slack_results:
                if isinstance(res, dict):
                    text = res.get("text", "")
                    url = res.get("metadata", {}).get("permalink", "")
                    unique_key = url or text
                else:
                    text = res
                    url = ""
                    unique_key = text
                if unique_key in seen:
                    continue
                seen.add(unique_key)
                slack_docs.append({
                    "title": "Slack Message",
                    "url": url,
                    "content": text,
                    "source": "slack"
                })

        # --- MCP search (only for metric queries) ---
        mcp_docs = []
        if ENABLE_MCP and is_metric_query(user_query):
            mcp_result = mcp_client.run_agentic_inference(user_query)
            # Format the answer and reasoning steps as context docs
            if "answer" in mcp_result:
                mcp_docs.append({
                    "title": "MCP Metric Answer",
                    "url": "",
                    "content": mcp_result["answer"],
                    "source": "mcp"
                })
            if "reasoning_steps" in mcp_result:
                for step in mcp_result["reasoning_steps"]:
                    mcp_docs.append({
                        "title": f"MCP Reasoning Step {step.get('id', '')} ({step.get('tool', '')})",
                        "url": "",
                        "content": step.get("response", ""),
                        "source": "mcp"
                    })

        # --- Aggregate all docs ---
        all_docs = mcp_docs + slack_docs + typesense_docs

        if not all_docs:
            print("Omni: No relevant documents found.\n")
        else:
            answer = synthesize_answer(user_query, all_docs)
            print("\nOmni:", answer)
            print("\n" + "="*40 + "\n")
                