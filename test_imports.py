#!/usr/bin/env python3
"""
Test script to verify import shims work correctly.
Run this to check if all modules can be imported successfully.
"""

def test_imports():
    """Test all import shims."""
    print("Testing import shims...")
    
    try:
        from import_shims import (
            get_llm, 
            KeywordExtractor, 
            prune_stopwords_from_results,
            run_chunking,
            SlackSearcher,
            MCPRegistry,
            validate_imports
        )
        print("✅ All imports successful")
        
        # Test validation
        status = validate_imports()
        print(f"Module status: {status}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1) 