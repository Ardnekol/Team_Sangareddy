#!/usr/bin/env python
"""
Verification script to check if all dependencies are correctly installed.
"""

import sys

def check_imports():
    """Check if all required imports work."""
    errors = []
    
    print("Checking dependencies...")
    print("=" * 60)
    
    # Check transformers
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
        
        # Check specific function
        from transformers.pytorch_utils import translate_to_torch_parallel_style
        print("✅ translate_to_torch_parallel_style: Available")
        
        # is_quanto_available is optional, not critical
        try:
            from transformers.utils import is_quanto_available
            print("✅ is_quanto_available: Available")
        except ImportError:
            print("⚠️  is_quanto_available: Not available (optional, not critical)")
    except ImportError as e:
        errors.append(f"❌ transformers: {e}")
        print(f"❌ transformers: {e}")
    
    # Check sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers: Available")
    except ImportError as e:
        errors.append(f"❌ sentence-transformers: {e}")
        print(f"❌ sentence-transformers: {e}")
    
    # Check tokenizers
    try:
        import tokenizers
        print(f"✅ tokenizers: {tokenizers.__version__}")
    except ImportError as e:
        errors.append(f"❌ tokenizers: {e}")
        print(f"❌ tokenizers: {e}")
    
    # Check timm
    try:
        import timm
        print(f"✅ timm: {timm.__version__}")
    except ImportError as e:
        errors.append(f"❌ timm: {e}")
        print(f"❌ timm: {e}")
    
    # Check vector store
    try:
        from vector_store import TicketVectorStore
        print("✅ vector_store: Available")
        
        # Try to initialize
        vs = TicketVectorStore()
        print("✅ Vector store initialization: Success")
    except Exception as e:
        errors.append(f"❌ vector_store: {e}")
        print(f"❌ vector_store: {e}")
    
    print("=" * 60)
    
    if errors:
        print("\n❌ Some dependencies have issues. Please run:")
        print("   pip install transformers==4.48.0 tokenizers==0.21.4 sentence-transformers==2.7.0 timm>=1.0.0")
        return False
    else:
        print("\n✅ All dependencies are correctly installed!")
        return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)

