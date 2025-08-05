#!/bin/bash
set -e

# This script is run from omni_gpt/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPLIED_AI_DIR="$SCRIPT_DIR/applied_ai"

if [ -d "./venv" ]; then
  echo "Activating local virtual environment..."
  source ./venv/bin/activate
else
  echo "❌ No venv found at ./venv. Please create one with 'python -m venv venv' first."
  exit 1
fi

echo "📦 Installing applied-ai libraries in editable mode..."

pip install -e "$APPLIED_AI_DIR/chunking"
pip install -e "$APPLIED_AI_DIR/retrieval"
pip install -e "$APPLIED_AI_DIR/generation"
pip install -e "$APPLIED_AI_DIR/mcp_client"
pip install -e "$APPLIED_AI_DIR/keyword_extraction"
pip install -e "$APPLIED_AI_DIR/slack_search"

echo "✅ All applied-ai packages installed in editable mode."
