#!/bin/bash
# Replit setup script - Run this once after cloning

echo "🚀 Setting up GenAI Ticket Analysis System on Replit..."
echo ""

# Check if index exists
if [ -f "ticket_index.faiss" ] && [ -f "ticket_index_metadata.pkl" ]; then
    echo "✅ Index files found - skipping build"
else
    echo "📦 Building vector index (this will take 20-30 minutes)..."
    echo "   This is a one-time setup. Please be patient."
    python setup.py
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "   streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
echo ""
echo "Or click the 'Run' button in Replit."

