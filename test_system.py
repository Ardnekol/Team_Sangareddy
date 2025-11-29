"""
Test script to verify the system works correctly.
"""

import os
from data_processor import TicketDataProcessor
from vector_store import TicketVectorStore


def test_system():
    """Test the vector store and retrieval."""
    print("Testing Ticket Analysis System...")
    print("=" * 60)
    
    # Test data loading
    print("\n1. Testing data loading...")
    data_path = 'telecom_tickets_10000_12cats.json'
    processor = TicketDataProcessor(data_path)
    tickets = processor.load_tickets()
    assert len(tickets) > 0, "No tickets loaded!"
    print(f"   ✅ Loaded {len(tickets)} tickets")
    
    # Test vector store
    print("\n2. Testing vector store...")
    vector_store = TicketVectorStore()
    
    if vector_store.load_index():
        print("   ✅ Index loaded successfully")
    else:
        print("   ⚠️  Index not found. Run setup.py first or initialize from Streamlit app.")
        return
    
    # Test search
    print("\n3. Testing similarity search...")
    test_query = "Customer reports intermittent internet connectivity on mobile data"
    results = vector_store.search(test_query, k=3)
    
    assert len(results) > 0, "No search results!"
    print(f"   ✅ Found {len(results)} similar tickets")
    
    for i, (ticket, score) in enumerate(results, 1):
        print(f"   Result {i}: Similarity {score:.2%} - Category: {ticket.get('category')}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! System is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    test_system()

