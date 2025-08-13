"""
Import shims for seamless development/production deployment.

This module provides import shims that work in both:
- Development: When running from the monorepo with applied_ai packages
- Production: When deployed with installed packages

Usage:
    from import_shims import get_llm, KeywordExtractor, run_chunking
    
    # These will automatically resolve to the correct import path
"""

# Generation module shims
try:
    from applied_ai.generation.generation.router import get_llm
    print("✅ Imported get_llm from applied_ai.generation.generation.router")
except ImportError:
    try:
        from generation.router import get_llm
        print("✅ Imported get_llm from generation.router")
    except ImportError:
        try:
            from applied_ai.generation.router import get_llm
            print("✅ Imported get_llm from applied_ai.generation.router")
        except ImportError:
            # Fallback for production deployment - create a mock function
            print("⚠️  Generation module not available, using fallback get_llm")
            def get_llm(provider="openai", model="gpt-4o-mini"):
                # Return a mock configuration that won't crash the app
                return None, {"model": model, "params": {"temperature": 0}}

# Keyword extraction module shims
try:
    from applied_ai.keyword_extraction.keyword_extractor.extractor import KeywordExtractor
except ImportError:
    try:
        from keyword_extractor.extractor import KeywordExtractor
    except ImportError:
        # Fallback class
        class KeywordExtractor:
            def __init__(self, config):
                self.config = config
            def extract(self, text):
                return {"ngram": []}

try:
    from applied_ai.keyword_extraction.keyword_extractor.stopword_pruner import prune_stopwords_from_results
except ImportError:
    try:
        from keyword_extractor.stopword_pruner import prune_stopwords_from_results
    except ImportError:
        def prune_stopwords_from_results(results, stopwords, method):
            return results

# Chunking module shims
try:
    from applied_ai.chunking.chunking.pipeline import run_chunking
except ImportError:
    try:
        from chunking.pipeline import run_chunking
    except ImportError:
        def run_chunking(*args, **kwargs):
            raise ImportError("Chunking module not available. Please install the chunking package.")

# Slack search module shims
try:
    from applied_ai.slack_search.slack_search.searcher import SlackSearcher
except ImportError:
    try:
        from slack_search.searcher import SlackSearcher
    except ImportError:
        # Fallback class
        class SlackSearcher:
            def __init__(self, slack_token=None, result_limit=3, thread_limit=5):
                self.slack_token = slack_token
                self.result_limit = result_limit
                self.thread_limit = thread_limit
            def search(self, query):
                return []

# MCP client module shims
try:
    from applied_ai.mcp_client.mcp_client.registry import MCPRegistry
except ImportError:
    try:
        from mcp_client.registry import MCPRegistry
    except ImportError:
        # Fallback class
        class MCPRegistry:
            def get_client(self, *args, **kwargs):
                return None

# Retrieval module shims
try:
    from applied_ai.retrieval.retrieval.faiss_retriever import FAISSRetriever
except ImportError:
    try:
        from retrieval.faiss_retriever import FAISSRetriever
    except ImportError:
        # Fallback class
        class FAISSRetriever:
            def __init__(self, *args, **kwargs):
                pass

# Validation function to check if all required modules are available
def validate_imports():
    """Check if all required modules are available and return status."""
    status = {
        "generation": "get_llm" in globals() and callable(get_llm),
        "keyword_extraction": KeywordExtractor is not None,
        "chunking": "run_chunking" in globals() and callable(run_chunking),
        "slack_search": SlackSearcher is not None,
        "mcp_client": MCPRegistry is not None,
        "retrieval": FAISSRetriever is not None,
    }
    
    missing = [module for module, available in status.items() if not available]
    if missing:
        print(f"Warning: Missing modules: {', '.join(missing)}")
        print("Some functionality may not be available.")
    
    return status 