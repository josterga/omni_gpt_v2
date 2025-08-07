import os
import sys
sys.path.insert(0, os.path.abspath(".")) 
import time
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from applied_ai.generation.generation.router import get_llm
from applied_ai.keyword_extraction.keyword_extractor.extractor import KeywordExtractor
from applied_ai.keyword_extraction.keyword_extractor.stopword_pruner import prune_stopwords_from_results
from applied_ai.slack_search.slack_search.searcher import SlackSearcher
from applied_ai.mcp_client.mcp_client.registry import MCPRegistry
from applied_ai.retrieval.retrieval.faiss_retriever import FAISSRetriever
from applied_ai.chunking.chunking.pipeline import run_chunking
import faiss
import openai


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
            "q": "",
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
SLACK_TOKEN = os.getenv("SLACK_API_KEY")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

FAISS_COMMUNITY_INDEX_PATH = "/Users/james/Documents/projects/omni_gpt/faiss/ollama-paragraph-discourse.index"
FAISS_COMMUNITY_META_PATH = "/Users/james/Documents/projects/omni_gpt/faiss/ollama-paragraph-discourse.meta.json"

FAISS_DOCS_INDEX_PATH = "/Users/james/Documents/projects/omni_gpt/faiss/ollama-paragraph-docs.index"
FAISS_DOCS_META_PATH = "/Users/james/Documents/projects/omni_gpt/faiss/ollama-paragraph-docs.meta.json"


def load_json_embeddings(json_dir_or_file):
    all_chunks = []
    paths = [Path(json_dir_or_file)] if Path(json_dir_or_file).is_file() else Path(json_dir_or_file).glob("*.json")
    for path in paths:
        with open(path) as f:
            for chunk in json.load(f):
                if "embedding" in chunk and isinstance(chunk["embedding"], list):
                    all_chunks.append(chunk)
    return all_chunks

def search_json_chunks(query_embedding, chunks, top_k=5):
    if not chunks:
        return []
    embeddings = np.array([c["embedding"] for c in chunks])
    scores = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

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

slack_searcher = SlackSearcher(slack_token=SLACK_TOKEN, result_limit=5, thread_limit=5)

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



def fetch_live_content(url, timeout=8):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return None

def search_typesense_ngrams(ngrams, max_results=5):
    docs, seen_urls = [], set()
    ngram_list = ngrams.get("ngram") if isinstance(ngrams, dict) else ngrams
    for ngram in ngram_list:
        body = TYPESENSE_SEARCH_BODY_TEMPLATE.copy()
        body["searches"][0] = body["searches"][0].copy()
        body["searches"][0]["q"] = ngram
        try:
            resp = requests.post(TYPESENSE_URL, headers=TYPESENSE_HEADERS, data=json.dumps(body), timeout=8)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception:
            continue
        for group in results[0].get("grouped_hits", []):
            for hit in group.get("hits", []):
                doc = hit["document"]
                url = doc.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                content = fetch_live_content(url) or BeautifulSoup(doc.get("content", ""), "html.parser").get_text(" ", strip=True)
                title = " > ".join([doc.get(f"hierarchy.lvl{i}") for i in range(7) if doc.get(f"hierarchy.lvl{i}")])
                docs.append({"title": title, "url": url, "content": content})
                if len(docs) >= max_results:
                    return docs
                time.sleep(0.5)
    return docs

def is_metric_query(query):
    keywords = ["how many", "count", "average", "sum", "total", "report", "list all", "github", "users", "opportunities"]
    return any(kw in query.lower() for kw in keywords)

def synthesize_answer(query, docs, provider="openai", model="gpt-4o-mini"):
    context = "\n\n".join([f"{d['title']} ({d['url']}):\n{d['content']}" for d in docs])
    llm, cfg = get_llm(provider=provider, model=model)
    messages = [
        {"role": "system", "content": (
            "You are an AI assistant for Omni Analytics. Answer using ONLY the provided context, "
            "citing 'Slack', 'Documentation', or 'Community'."
        )},
        {"role": "user", "content": (
            f"**User Question**:\n{query}\n\n"
            f"**Available Information**:\n{context}\n\n"
            "Now write your answer."
        )}
    ]
    return llm.chat(messages, model=cfg["model"], **cfg.get("params", {}))

def handle_user_query(query):
    extractor = KeywordExtractor(ngram_config)
    ngrams = extractor.extract(query)
    ngrams = prune_stopwords_from_results(ngrams, set(ngram_config["ngram"]["stopwords"]), "remove_stopwords_within_phrase")

    # typesense_docs = search_typesense_ngrams(ngrams)
    # for doc in typesense_docs:
    #     doc["source"] = "typesense"

    slack_docs = []
    seen = set()
    slack_ngrams = ngrams.get("ngram") if isinstance(ngrams, dict) else ngrams

    for ng in slack_ngrams:
        for res in slack_searcher.search(ng):
            text = res.get("text", "") if isinstance(res, dict) else res
            url = res.get("metadata", {}).get("permalink", "") if isinstance(res, dict) else ""
            key = url or text
            if key in seen:
                continue
            seen.add(key)
            slack_docs.append({
                "title": "Slack Message",
                "url": url,
                "content": text,
                "source": "slack"
            })
    docs_chunks = load_json_embeddings("/Users/james/Documents/projects/omni_gpt/sources/docs/embeddings-000.json")
    community_chunks = load_json_embeddings("/Users/james/Documents/projects/omni_gpt/sources/discourse/discourse-000.json")

    query_chunks = run_chunking(
        raw_text=query,
        chunk_method="sentence",
        max_tokens=300,
        overlap_tokens=40,
        inject_headers=True,
        provider="ollama",
        model_name="nomic-embed-text"
    )
    query_embedding = query_chunks[0]["embedding"]

    top_docs = search_json_chunks(query_embedding, docs_chunks)
    top_docs_formatted = []
    for chunk in top_docs:
        top_docs_formatted.append({
            "title": chunk["metadata"].get("path", "Docs"),
            "url": "",
            "content": chunk.get("chunk_text", ""),
            "source": "docs"
        })
    top_community = search_json_chunks(query_embedding, community_chunks)
    top_community_formatted = []
    for chunk in top_community:
        top_community_formatted.append({
            "title": chunk["metadata"].get("path", "Docs"),
            "url": "",
            "content": chunk.get("chunk_text", ""),
            "source": "docs"
        })
    mcp_docs = []
    if ENABLE_MCP and is_metric_query(query):
        result = mcp_client.run_agentic_inference(query)
        if "answer" in result:
            mcp_docs.append({"title": "MCP Answer", "url": "", "content": result["answer"], "source": "mcp"})
        for step in result.get("reasoning_steps", []):
            mcp_docs.append({
                "title": f"MCP Step {step.get('id', '')}",
                "url": "",
                "content": step.get("response", ""),
                "source": "mcp"
            })

    all_docs = mcp_docs + slack_docs + top_docs_formatted + top_community_formatted #+ typesense_docs 
    if not all_docs:
        return "No relevant documents found.", []
    return synthesize_answer(query, all_docs), all_docs