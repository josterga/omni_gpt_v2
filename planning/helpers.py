import os, json, time
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

# -------- Embedding JSON helpers --------
def load_json_embeddings(json_dir_or_file):
    all_chunks = []
    paths = [Path(json_dir_or_file)] if Path(json_dir_or_file).is_file() else Path(json_dir_or_file).glob("*.json*")
    for path in paths:
        with open(path) as f:
            if path.suffix == ".jsonl":
                for line in f:
                    try:
                        chunk = json.loads(line)
                        if "embedding" in chunk and isinstance(chunk["embedding"], list):
                            all_chunks.append(chunk)
                    except json.JSONDecodeError:
                        continue
            elif path.suffix == ".json":
                try:
                    data = json.load(f)
                    for chunk in data:
                        if "embedding" in chunk and isinstance(chunk["embedding"], list):
                            all_chunks.append(chunk)
                except json.JSONDecodeError:
                    continue
    return all_chunks

def search_json_chunks(query_embedding, chunks, top_k=5):
    if not chunks:
        return []
    embeddings = np.array([c["embedding"] for c in chunks])
    scores = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# -------- Simple live fetch (Typesense fallback) --------
def fetch_live_content(url, timeout=8):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return None

# -------- Typesense (docs.omni.co) --------
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")
TYPESENSE_BASE_URL = os.getenv("TYPESENSE_BASE_URL")  # e.g. "xyz"
TYPESENSE_URL = f"https://{TYPESENSE_BASE_URL}-1.a1.typesense.net/multi_search?x-typesense-api-key={TYPESENSE_API_KEY}"
TYPESENSE_HEADERS = {"accept": "application/json, text/plain, */*", "content-type": "text/plain"}
TYPESENSE_BODY_TEMPLATE = {
    "searches": [{
        "collection": "omni-docs",
        "q": "",
        "query_by": "hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3,hierarchy.lvl4,hierarchy.lvl5,hierarchy.lvl6,content",
        "include_fields": "hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3,hierarchy.lvl4,hierarchy.lvl5,hierarchy.lvl6,content,anchor,url,type,id",
        "highlight_full_fields": "hierarchy.lvl0,hierarchy.lvl1,hierarchy.lvl2,hierarchy.lvl3,hierarchy.lvl4,hierarchy.lvl5,hierarchy.lvl6,content",
        "group_by": "url", "group_limit": 3, "sort_by": "item_priority:desc",
        "snippet_threshold": 8, "highlight_affix_num_tokens": 4, "filter_by": ""
    }]
}

def search_typesense_ngrams(ngrams, max_results=5):
    docs, seen = [], set()
    ngram_list = ngrams.get("ngram") if isinstance(ngrams, dict) else ngrams
    for ngram in ngram_list:
        body = json.loads(json.dumps(TYPESENSE_BODY_TEMPLATE))
        body["searches"][0]["q"] = ngram
        try:
            resp = requests.post(TYPESENSE_URL, headers=TYPESENSE_HEADERS, data=json.dumps(body), timeout=8)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception:
            continue
        for group in (results[0].get("grouped_hits", []) if results else []):
            for hit in group.get("hits", []):
                doc = hit["document"]; url = doc.get("url") or ""
                if url in seen: continue
                seen.add(url)
                content = fetch_live_content(url) or doc.get("content", "")
                title = " > ".join([doc.get(f"hierarchy.lvl{i}") for i in range(7) if doc.get(f"hierarchy.lvl{i}")])
                docs.append({"title": title, "url": url, "content": content})
                if len(docs) >= max_results:
                    return docs
                time.sleep(0.4)
    return docs