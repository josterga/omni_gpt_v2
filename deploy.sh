#!/bin/bash

echo "🚀 Deploying Omni-GPT Streamlit App..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this from the project root."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found."
    exit 1
fi

# Check if import_shims.py exists
if [ ! -f "import_shims.py" ]; then
    echo "❌ Error: import_shims.py not found."
    exit 1
fi

echo "✅ All required files found"

# Test imports
echo "🧪 Testing imports..."
python3 deployment_test.py
if [ $? -ne 0 ]; then
    echo "❌ Import tests failed. Please fix the issues before deploying."
    exit 1
fi

echo "✅ Import tests passed"

# Check if streamlit is available
echo "🔍 Checking Streamlit availability..."
python3 -c "import streamlit; print('✅ Streamlit available')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Streamlit not available. Please install it first:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo "🎉 Ready for deployment!"
echo ""
echo "To run the app locally:"
echo "   streamlit run main.py"
echo ""
echo "To deploy to Streamlit Cloud:"
echo "   1. Push this code to GitHub"
echo "   2. Connect your repo to Streamlit Cloud"
echo "   3. Set the main file path to: main.py"
echo ""
echo "Environment variables needed:"
echo "   - FATHOM_API_KEY: Your Fathom API key"
echo "   - OPENAI_API_KEY: Your OpenAI API key"
echo "   - SLACK_API_KEY: Your Slack API key (optional)"
echo "   - OMNI_MODEL_ID: Your Omni model ID (optional)"
echo "   - OMNI_API_KEY: Your Omni API key (optional)"
echo "   - BASE_URL: Your base URL (optional)"
echo "   - TYPESENSE_API_KEY: Your Typesense API key (optional)"
echo "   - TYPESENSE_BASE_URL: Your Typesense base URL (optional)" 