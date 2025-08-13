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
except ImportError:
    try:
        from generation.generation.router import get_llm
    except ImportError:
        # Fallback for production deployment
        def get_llm(provider="openai", model="gpt-4o-mini"):
            raise ImportError("Generation module not available. Please install the generation package.")

# Keyword extraction module shims
try:
    from applied_ai.keyword_extraction.keyword_extractor.extractor import KeywordExtractor
except ImportError:
    try:
        from keyword_extraction.keyword_extractor.extractor import KeywordExtractor
    except ImportError:
        KeywordExtractor = None

try:
    from applied_ai.keyword_extraction.keyword_extractor.stopword_pruner import prune_stopwords_from_results
except ImportError:
    try:
        from keyword_extraction.keyword_extractor.stopword_pruner import prune_stopwords_from_results
    except ImportError:
        def prune_stopwords_from_results(results, stopwords, method):
            return results

# Chunking module shims
try:
    from applied_ai.chunking.chunking.pipeline import run_chunking
except ImportError:
    try:
        from chunking.chunking.pipeline import run_chunking
    except ImportError:
        def run_chunking(*args, **kwargs):
            raise ImportError("Chunking module not available. Please install the chunking package.")

# Slack search module shims
try:
    from applied_ai.slack_search.slack_search.searcher import SlackSearcher
except ImportError:
    try:
        from slack_search.slack_search.searcher import SlackSearcher
    except ImportError:
        SlackSearcher = None

# MCP client module shims
try:
    from applied_ai.mcp_client.mcp_client.registry import MCPRegistry
except ImportError:
    try:
        from mcp_client.mcp_client.registry import MCPRegistry
    except ImportError:
        MCPRegistry = None

# Retrieval module shims
try:
    from applied_ai.retrieval.retrieval.faiss_retriever import FAISSRetriever
except ImportError:
    try:
        from retrieval.retrieval.faiss_retriever import FAISSRetriever
    except ImportError:
        FAISSRetriever = None

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