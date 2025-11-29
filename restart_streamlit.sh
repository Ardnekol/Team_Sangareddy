#!/bin/bash
# Script to properly restart Streamlit with cleared cache

echo "Stopping any running Streamlit instances..."
pkill -f streamlit || true

echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "Clearing Streamlit cache..."
rm -rf .streamlit/cache 2>/dev/null || true

echo ""
echo "✅ Cache cleared. Now start Streamlit with:"
echo "   streamlit run app.py"
echo ""
echo "If you still see errors, try:"
echo "   1. Close all Streamlit browser tabs"
echo "   2. Wait a few seconds"
echo "   3. Run: streamlit run app.py --server.headless true"

