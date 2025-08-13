#!/usr/bin/env python3
"""
Deployment test script for Streamlit app.
This tests that all components can be imported and initialized correctly.
"""

def test_basic_imports():
    """Test basic module imports."""
    print("Testing basic imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit")
        
        from import_shims import validate_imports
        print("✅ Import shims")
        
        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_core_modules():
    """Test core module imports."""
    print("Testing core modules...")
    
    try:
        # Test import shims
        from import_shims import (
            get_llm, 
            KeywordExtractor, 
            prune_stopwords_from_results,
            run_chunking,
            SlackSearcher,
            MCPRegistry,
            FAISSRetriever
        )
        print("✅ All import shims")
        
        # Test validation
        from import_shims import validate_imports
        status = validate_imports()
        print(f"✅ Module status: {status}")
        
        return True
    except ImportError as e:
        print(f"❌ Core module import failed: {e}")
        return False

def test_planning_modules():
    """Test planning module imports."""
    print("Testing planning modules...")
    
    try:
        from planning.types import ToolCatalog, PlanStep, ToolResult
        print("✅ Planning types")
        
        from planning.planner import ToolPlanner
        print("✅ Tool planner")
        
        from planning.executor import ToolExecutor
        print("✅ Tool executor")
        
        return True
    except ImportError as e:
        print(f"❌ Planning module import failed: {e}")
        return False

def test_orchestrators():
    """Test orchestrator imports."""
    print("Testing orchestrators...")
    
    try:
        from orchestrators.direct import run as direct_run
        print("✅ Direct orchestrator")
        
        from orchestrators.planned import run as planned_run
        print("✅ Planned orchestrator")
        
        return True
    except ImportError as e:
        print(f"❌ Orchestrator import failed: {e}")
        return False

def test_tooling():
    """Test tooling imports."""
    print("Testing tooling...")
    
    try:
        from tooling.common_utils import make_docs_url_from_path
        print("✅ Common utils")
        
        from tooling.query_artifacts import LazyQueryArtifacts
        print("✅ Query artifacts")
        
        return True
    except ImportError as e:
        print(f"❌ Tooling import failed: {e}")
        return False

def test_fathom_module():
    """Test fathom module imports."""
    print("Testing fathom module...")
    
    try:
        from fathom_module import fathom_api
        print("✅ Fathom API")
        
        return True
    except ImportError as e:
        print(f"❌ Fathom module import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting deployment tests...\n")
    
    tests = [
        test_basic_imports,
        test_core_modules,
        test_planning_modules,
        test_orchestrators,
        test_tooling,
        test_fathom_module,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 