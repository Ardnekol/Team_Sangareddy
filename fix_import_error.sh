#!/bin/bash
# Script to fix the ImportError with transformers and sentence-transformers

echo "Fixing transformers/sentence-transformers compatibility issue..."
echo ""

# Uninstall conflicting packages
echo "Step 1: Uninstalling conflicting packages..."
pip uninstall -y transformers sentence-transformers tokenizers

# Install compatible versions (tested and working)
echo ""
echo "Step 2: Installing compatible versions..."
pip install transformers==4.48.0
pip install tokenizers==0.21.4
pip install sentence-transformers==2.7.0
pip install timm>=1.0.0
pip install "pillow>=7.1.0,<11"

echo ""
echo "✅ Done! Try running the app again: streamlit run app.py"

